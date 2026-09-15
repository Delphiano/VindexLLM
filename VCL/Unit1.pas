unit Unit1;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, VindexLLM.Inference, VindexLLM.Sampler,
  VindexLLM.TokenWriter, VindexLLM.Utils, UTest.Model.Gemma3, VindexLLM.Model.Llama,
  IOUtils, System.Generics.Collections;

type
  TForm1 = class(TForm)
    Button1: TButton;
    Memo1: TMemo;
    Button2: TButton;
    Button3: TButton;
    mPrompt: TMemo;
    Edit1: TEdit;
    procedure Button1Click(Sender: TObject);
    procedure Button2Click(Sender: TObject);
    procedure Button3Click(Sender: TObject);
  private
    LInference: TVdxInference;
    GTokenWriter: TVdxTokenWriter;
    { Private declarations }
  public
    { Public declarations }
  end;

var
  Form1: TForm1;

implementation

{$R *.dfm}

procedure StatusCallback(const AText: string; const AUserData: Pointer);
begin
  Form1.Memo1.Lines.Add(AText);
end;

procedure PrintToken(const AToken: string; const AUserData: Pointer);
begin
  Form1.Memo1.Lines.Text := Form1.Memo1.Lines.Text + AToken;
  Application.ProcessMessages;
end;

function CancelCallback(const AUserData: Pointer): Boolean;
begin
  Result := False;
  //Form1.Memo1.Lines.Append('Cancelado');
end;

procedure InferenceEventCallback(const AEvent: TVdxInferenceEvent; const AUserData: Pointer);
begin
  //Form1.Memo1.Lines.Append('Evento novo');
end;

procedure PrintErrors(const AInference: TVdxInference);
var
  LErrors: TVdxErrors;
  LItems: TList<TVdxError>;
  LI: Integer;
  LErr: TVdxError;
  LColor: string;
  LLabel: string;
begin
  LErrors := AInference.GetErrors();
  if LErrors = nil then
    Exit;
  LItems := LErrors.GetItems();
  if LItems.Count = 0 then
    Exit;

  TVdxUtils.PrintLn('');
  for LI := 0 to LItems.Count - 1 do
  begin
    LErr := LItems[LI];
    case LErr.Severity of
      esHint:
      begin
        LColor := COLOR_CYAN;
        LLabel := 'HINT';
      end;
      esWarning:
      begin
        LColor := COLOR_YELLOW;
        LLabel := 'WARN';
      end;
      esError:
      begin
        LColor := COLOR_RED;
        LLabel := 'ERROR';
      end;
      esFatal:
      begin
        LColor := COLOR_MAGENTA;
        LLabel := 'FATAL';
      end;
    else
      LColor := COLOR_WHITE;
      LLabel := '?';
    end;

    if LErr.Code <> '' then
      Form1.Memo1.Lines.Append(LErr.Message)
    else
      Form1.Memo1.Lines.Append(LErr.Message);
  end;
end;

procedure TForm1.Button1Click(Sender: TObject);
var
  LConfig: TVdxSamplerConfig;
  LLoaded: Boolean;
begin
  // --- Step 0: Create the token writer for word-wrapped streaming output ---
  GTokenWriter := TVdxTokenWriter.Create();
  GTokenWriter.MaxWidth := 118;

  // --- Step 1: Create the inference engine ---
  // TVdxInference is the main orchestrator. It owns all subsystems (Vulkan
  // compute, attention, norms, tokenizer, sampler) and manages their lifecycle.
  LInference := TVdxInference.Create();
  // --- Step 2: Register callbacks ---
  // StatusCallback:  receives progress messages during model loading
  // PrintToken:      streams each generated token to console in real time
  // EventCallback:   notifies on prefill/generate start/end transitions
  // CancelCallback:  polled per-layer; return True (ESC key) to abort
  LInference.SetStatusCallback(StatusCallback, nil);
  LInference.SetTokenCallback(PrintToken, nil);
  LInference.SetInferenceEventCallback(InferenceEventCallback, nil);
  LInference.SetCancelCallback(CancelCallback, nil);

  // --- Step 3: Load the model ---
  // This is the heavy operation. It will:
  //   - Memory-map the GGUF file (zero-copy access to weight data)
  //   - Detect architecture from GGUF metadata (must be "gemma3")
  //   - Read model dimensions (layers, hidden_dim, ffn_width, head counts)
  //   - Initialize Vulkan (find GPU, create device, compute queue)
  //   - Create all 30 compute shader pipelines from embedded SPIR-V
  //   - Upload ~4-8 GB of weights to GPU VRAM via staging buffers
  //   - Allocate TQ3-compressed KV cache (10.7x smaller than F32)
  //   - Load BPE tokenizer vocabulary directly from GGUF
  //   - Report total VRAM usage via status callback
  // Returns False if anything fails (wrong architecture, missing tensors, etc.)
  LLoaded := LInference.LoadModel(Edit1.Text, 4096);

  // Check for errors from model loading (architecture mismatch, missing
  // tensors, Vulkan init failure, etc.)
  PrintErrors(LInference);
  if not LLoaded then
    Exit;

  // --- Step 4: Configure the token sampler ---
  // Start from defaults (Temperature=0 = greedy argmax) then override
  // with recommended settings for this specific model variant.
  // These values are what is recommend by Google for gemma-3-4b-it:
  LConfig := TVdxSampler.DefaultConfig();
  LConfig.Temperature := 1.0;
  LConfig.TopK := 64;
  LConfig.TopP := 0.95;
  Lconfig.MinP := 0.0;
  LConfig.RepeatPenalty := 1.0;
  LConfig.RepeatWindow := 64;
  LConfig.Seed := 0;
  LInference.SetSamplerConfig(LConfig);


  // Check for errors from generation (prompt too long, context overflow, etc.)
  //PrintErrors(LInference);

  // --- Step 6: Print performance stats ---
  // Shows prefill throughput, generation throughput, TTFT, stop reason,
  // and VRAM usage breakdown (weights, KV cache, work buffers)
  //PrintStats(LInference.GetStats());
end;

procedure TForm1.Button2Click(Sender: TObject);
begin
  // --- Step 7: Unload model ---
  // Frees all GPU resources: shader pipelines, descriptor sets/pools,
  // weight buffers, KV cache, work buffers, embedding table copy.
  // Closes the memory-mapped GGUF file. Destroys all subsystem objects.
  // After this call, the inference engine can load a different model.
  LInference.UnloadModel();

  // Destroy the inference engine itself
  LInference.Free();

  // Destroy the token writer
  GTokenWriter.Free();
  GTokenWriter := nil;
end;

procedure TForm1.Button3Click(Sender: TObject);
begin
  // --- Step 5: Generate text ---
  // Generate() does the full pipeline:
  //   1. Format prompt with Gemma 3 chat template
  //   2. Tokenize (BPE encode with BOS token)
  //   3. Batched prefill — all prompt tokens in parallel through 34 layers
  //   4. Autoregressive generation — one token at a time, up to 1024 tokens
  //   5. Each token is decoded and sent to PrintToken callback (streaming output)
  //   6. Stops when: EOS, <end_of_turn>, 1024 tokens reached, context full, or ESC
  // The return value is the complete generated string (same text that was streamed).

  var LConfig: TVdxSamplerConfig;
  LConfig := TVdxSampler.DefaultConfig;

  {LConfig.Temperature := 0.2;   // 0 = determinístico (greedy)
  LConfig.TopK := 40;           // 0 = desabilitado
  LConfig.TopP := 0.90;         // 1.0 = desabilitado
  LConfig.MinP := 0.05;         // 0 = desabilitado
  LConfig.RepeatPenalty := 1.1; // 1.0 = desabilitado
  LConfig.RepeatWindow := 64;   // tokens analisados para repetição
  LConfig.Seed := 1234;         // 0 = aleatório; outro valor = reproduzível }

  LConfig.Temperature := 0.7;
  LConfig.TopK := 40;
  LConfig.TopP := 0.90;
  LConfig.MinP := 0.05;
  LConfig.RepeatPenalty := 1.1;
  LConfig.RepeatWindow := 64;
  LConfig.Seed := 0;

  LInference.SetSamplerConfig(LConfig);

  TThread.CreateAnonymousThread(procedure
  begin
    GTokenWriter.Reset();
    //LInference.ResetKVCache;
    LInference.Generate(mPrompt.Text, 50);
  end).Start;
end;

end.


