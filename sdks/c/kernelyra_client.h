#ifndef KERNELYRA_CLIENT_H
#define KERNELYRA_CLIENT_H

#include <stddef.h>
#include <stdio.h>
#include <string.h>

/* The host owns the persistent kernelyra-rpc process and supplies one-line I/O. */
typedef int (*kernelyra_transport)(const char *request, char *response, size_t capacity, void *context);

typedef struct {
    kernelyra_transport transport;
    void *context;
    unsigned long long request_id;
} kernelyra_client;

static int kernelyra_escape(const char *value, char *output, size_t capacity) {
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

static int kernelyra_call(kernelyra_client *client, const char *method, const char *params_json,
                          char *response, size_t response_capacity) {
    char request[65536];
    int written = snprintf(request, sizeof(request),
        "{\"id\":%llu,\"method\":\"%s\",\"params\":%s}\n",
        ++client->request_id, method, params_json ? params_json : "{}");
    if (!client->transport || written < 0 || (size_t) written >= sizeof(request)) return -1;
    return client->transport(request, response, response_capacity, client->context);
}

static int kernelyra_plan(kernelyra_client *client, const char *dataset, char *response, size_t capacity) {
    char escaped[32768], params[33024];
    if (kernelyra_escape(dataset, escaped, sizeof(escaped)) != 0) return -1;
    if (snprintf(params, sizeof(params), "{\"dataset\":\"%s\"}", escaped) < 0) return -1;
    return kernelyra_call(client, "plan", params, response, capacity);
}

static int kernelyra_train(kernelyra_client *client, const char *dataset, char *response, size_t capacity) {
    char escaped[32768], params[33024];
    if (kernelyra_escape(dataset, escaped, sizeof(escaped)) != 0) return -1;
    if (snprintf(params, sizeof(params), "{\"dataset\":\"%s\"}", escaped) < 0) return -1;
    return kernelyra_call(client, "train", params, response, capacity);
}

static int kernelyra_finetune(kernelyra_client *client, const char *model, const char *dataset,
                              char *response, size_t capacity) {
    char escaped_model[16384], escaped_dataset[16384], params[33024];
    if (kernelyra_escape(model, escaped_model, sizeof(escaped_model)) != 0) return -1;
    if (kernelyra_escape(dataset, escaped_dataset, sizeof(escaped_dataset)) != 0) return -1;
    if (snprintf(params, sizeof(params), "{\"model\":\"%s\",\"dataset\":\"%s\"}",
                 escaped_model, escaped_dataset) < 0) return -1;
    return kernelyra_call(client, "finetune", params, response, capacity);
}

#endif
