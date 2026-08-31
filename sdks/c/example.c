#include "kernelyra_client.h"

#include <string.h>

static int mock_transport(const char *request, char *response, size_t capacity, void *context) {
    const char *required[] = {
        "\"execution\":\"cpu\"",
        "\"threads\":8"
    };
    size_t index;
    (void) context;
    for (index = 0; index < sizeof(required) / sizeof(required[0]); ++index) {
        if (!strstr(request, required[index])) return -1;
    }
    return snprintf(response, capacity, "{\"ok\":true,\"result\":{}}\n") < 0 ? -1 : 0;
}

int main(void) {
    char response[128];
    kernelyra_client client = {mock_transport, NULL, 0};
    kernelyra_run_options options = {0};
    options.target = "label";
    options.execution = KERNELYRA_EXECUTION_CPU;
    options.cpu = 90;
    options.ram = 80;
    options.threads = 8;
    return kernelyra_train_with_options(&client, "train.csv", &options, response, sizeof(response));
}
