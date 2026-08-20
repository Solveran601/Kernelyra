<?php
declare(strict_types=1);

final class KernelyraClient {
    private $process;
    private array $pipes;
    private int $id = 0;

    public function __construct(string $workspace, string $executable = 'kernelyra') {
        $command = [$executable, '--workspace', $workspace, 'rpc'];
        $this->process = proc_open($command, [['pipe','r'], ['pipe','w'], ['pipe','w']], $this->pipes);
        if (!is_resource($this->process)) throw new RuntimeException('Cannot start Kernelyra');
        $ready = json_decode((string) fgets($this->pipes[1]), true, flags: JSON_THROW_ON_ERROR);
        if (($ready['protocol'] ?? '') !== 'kernelyra-jsonl/1') throw new RuntimeException('Incompatible protocol');
    }

    public function call(string $method, array $params = []): array {
        $request = ['id' => ++$this->id, 'method' => $method, 'params' => $params];
        fwrite($this->pipes[0], json_encode($request, JSON_THROW_ON_ERROR) . "\n"); fflush($this->pipes[0]);
        $response = json_decode((string) fgets($this->pipes[1]), true, flags: JSON_THROW_ON_ERROR);
        if (!($response['ok'] ?? false)) throw new RuntimeException((string) ($response['error'] ?? 'Kernelyra error'));
        return $response['result'];
    }

    public function plan(string $dataset, array $options = []): array { return $this->call('plan', ['dataset'=>$dataset] + $options); }
    public function train(string $dataset, array $options = []): array { return $this->call('train', ['dataset'=>$dataset] + $options); }
    public function finetune(string $model, string $dataset, array $options = []): array { return $this->call('finetune', ['model'=>$model, 'dataset'=>$dataset] + $options); }
    public function close(): void { fclose($this->pipes[0]); fclose($this->pipes[1]); fclose($this->pipes[2]); proc_close($this->process); }
}
