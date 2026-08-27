#ifndef KERNELYRA_CLIENT_H
#define KERNELYRA_CLIENT_H

#include <stddef.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

/* The host owns the persistent kernelyra-rpc process and supplies one-line I/O. */
typedef int (*kernelyra_transport)(const char *request, char *response, size_t capacity, void *context);

typedef struct {
    kernelyra_transport transport;
    void *context;
    unsigned long long request_id;
} kernelyra_client;

/* Explicit execution vocabulary shared by the native SDKs. */
#define KERNELYRA_EXECUTION_AUTO "auto"
#define KERNELYRA_EXECUTION_CPU "cpu"
#define KERNELYRA_EXECUTION_HYBRID "hybrid"
#define KERNELYRA_PACK_CAREFUL "careful"
#define KERNELYRA_PACK_BALANCED "balanced"
#define KERNELYRA_PACK_THROUGHPUT "throughput"
#define KERNELYRA_PACK_MAXIMUM "maximum"

typedef struct {
    const char *target;
    const char *task;
    const char *backend;
    const char *execution;
    const char *algorithm_pack;
    unsigned cpu;
    unsigned ram;
    unsigned gpu;
    unsigned threads;
    unsigned long long max_steps;
    double target_metric;
} kernelyra_run_options;

static inline int kernelyra_escape(const char *value, char *output, size_t capacity) {
    size_t used = 0;
    for (; *value; ++value) {
        const char *escaped = NULL;
        char pair[3] = {'\\', 0, 0};
        if (*value == '\\' || *value == '"') { pair[1] = *value; escaped = pair; }
        else if (*value == '\n') escaped = "\\n";
        else if (*value == '\r') escaped = "\\r";
        else if (*value == '\t') escaped = "\\t";
        if (escaped) {
            size_t length = strlen(escaped);
            if (used + length >= capacity) return -1;
            memcpy(output + used, escaped, length); used += length;
        } else {
            if (used + 1 >= capacity) return -1;
            output[used++] = *value;
        }
    }
    if (used >= capacity) return -1;
    output[used] = 0;
    return 0;
}

static inline int kernelyra_call(kernelyra_client *client, const char *method, const char *params_json,
                          char *response, size_t response_capacity) {
    char request[65536];
    int written = snprintf(request, sizeof(request),
        "{\"id\":%llu,\"method\":\"%s\",\"params\":%s}\n",
        ++client->request_id, method, params_json ? params_json : "{}");
    if (!client->transport || written < 0 || (size_t) written >= sizeof(request)) return -1;
    return client->transport(request, response, response_capacity, client->context);
}

static inline int kernelyra_append(char *output, size_t capacity, size_t *used, const char *format, ...) {
    va_list arguments;
    int written;
    if (*used >= capacity) return -1;
    va_start(arguments, format);
    written = vsnprintf(output + *used, capacity - *used, format, arguments);
    va_end(arguments);
    if (written < 0 || (size_t) written >= capacity - *used) return -1;
    *used += (size_t) written;
    return 0;
}

static inline int kernelyra_append_string_field(char *output, size_t capacity, size_t *used,
                                          int *first, const char *name, const char *value) {
    char escaped[32768];
    if (!value || !*value) return 0;
    if (kernelyra_escape(value, escaped, sizeof(escaped)) != 0) return -1;
    if (kernelyra_append(output, capacity, used, "%s\"%s\":\"%s\"", *first ? "" : ",", name, escaped) != 0) return -1;
    *first = 0;
    return 0;
}

static inline int kernelyra_append_unsigned_field(char *output, size_t capacity, size_t *used,
                                            int *first, const char *name, unsigned long long value) {
    if (!value) return 0;
    if (kernelyra_append(output, capacity, used, "%s\"%s\":%llu", *first ? "" : ",", name, value) != 0) return -1;
    *first = 0;
    return 0;
}

/* Serialize only explicit options. A zero numeric value means "leave it to the engine". */
static inline int kernelyra_run_params(const char *dataset, const kernelyra_run_options *options,
                                char *params, size_t capacity) {
    size_t used = 0;
    int first = 1;
    if (!dataset || kernelyra_append(params, capacity, &used, "{") != 0) return -1;
    if (kernelyra_append_string_field(params, capacity, &used, &first, "dataset", dataset) != 0) return -1;
    if (options) {
        if (kernelyra_append_string_field(params, capacity, &used, &first, "target", options->target) != 0 ||
            kernelyra_append_string_field(params, capacity, &used, &first, "task", options->task) != 0 ||
            kernelyra_append_string_field(params, capacity, &used, &first, "backend", options->backend) != 0 ||
            kernelyra_append_string_field(params, capacity, &used, &first, "execution", options->execution) != 0 ||
            kernelyra_append_string_field(params, capacity, &used, &first, "algorithm_pack", options->algorithm_pack) != 0 ||
            kernelyra_append_unsigned_field(params, capacity, &used, &first, "cpu", options->cpu) != 0 ||
            kernelyra_append_unsigned_field(params, capacity, &used, &first, "ram", options->ram) != 0 ||
            kernelyra_append_unsigned_field(params, capacity, &used, &first, "gpu", options->gpu) != 0 ||
            kernelyra_append_unsigned_field(params, capacity, &used, &first, "threads", options->threads) != 0 ||
            kernelyra_append_unsigned_field(params, capacity, &used, &first, "max_steps", options->max_steps) != 0) return -1;
        if (options->target_metric > 0.0) {
            if (kernelyra_append(params, capacity, &used, "%s\"target_metric\":%.17g", first ? "" : ",", options->target_metric) != 0) return -1;
            first = 0;
        }
    }
    return kernelyra_append(params, capacity, &used, "}");
}

static inline int kernelyra_plan(kernelyra_client *client, const char *dataset, char *response, size_t capacity) {
    char params[65536];
    if (kernelyra_run_params(dataset, NULL, params, sizeof(params)) != 0) return -1;
    return kernelyra_call(client, "plan", params, response, capacity);
}

static inline int kernelyra_plan_with_options(kernelyra_client *client, const char *dataset,
                                       const kernelyra_run_options *options,
                                       char *response, size_t capacity) {
    char params[65536];
    if (kernelyra_run_params(dataset, options, params, sizeof(params)) != 0) return -1;
    return kernelyra_call(client, "plan", params, response, capacity);
}

static inline int kernelyra_train(kernelyra_client *client, const char *dataset, char *response, size_t capacity) {
    char params[65536];
    if (kernelyra_run_params(dataset, NULL, params, sizeof(params)) != 0) return -1;
    return kernelyra_call(client, "train", params, response, capacity);
}

static inline int kernelyra_train_with_options(kernelyra_client *client, const char *dataset,
                                        const kernelyra_run_options *options,
                                        char *response, size_t capacity) {
    char params[65536];
    if (kernelyra_run_params(dataset, options, params, sizeof(params)) != 0) return -1;
    return kernelyra_call(client, "train", params, response, capacity);
}

static inline int kernelyra_finetune(kernelyra_client *client, const char *model, const char *dataset,
                              char *response, size_t capacity) {
    char escaped_model[16384], escaped_dataset[16384], params[33024];
    if (kernelyra_escape(model, escaped_model, sizeof(escaped_model)) != 0) return -1;
    if (kernelyra_escape(dataset, escaped_dataset, sizeof(escaped_dataset)) != 0) return -1;
    if (snprintf(params, sizeof(params), "{\"model\":\"%s\",\"dataset\":\"%s\"}",
                 escaped_model, escaped_dataset) < 0) return -1;
    return kernelyra_call(client, "finetune", params, response, capacity);
}

#endif
