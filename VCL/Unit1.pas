unit Unit1;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants,
  System.Classes, System.SyncObjs, System.IOUtils, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.ComCtrls,
  VindexLLM.Session, VindexLLM.Sampler, FireDAC.VCLUI.Wait;

type
  TForm1 = class(TForm)
    Button1: TButton;
    Memo1: TMemo;
    Button2: TButton;
    Button3: TButton;
    mPrompt: TMemo;
    Edit1: TEdit;
    edTemperatura: TEdit;
    Label1: TLabel;
    Label2: TLabel;
    edTopK: TEdit;
    edTopP: TEdit;
    Label3: TLabel;
    Label4: TLabel;
    edMinP: TEdit;
    Label5: TLabel;
    edRepeatPenalty: TEdit;
    Memo2: TMemo;
    edMaxTokens: TEdit;
    Label6: TLabel;
    Label7: TLabel;
    edRagFile: TEdit;
    btnIndexRagFile: TButton;
    chkUseRag: TCheckBox;
    chkLoadExistingRag: TCheckBox;
    ProgressBar1: TProgressBar;
    OpenDialog1: TOpenDialog;
    procedure Button1Click(Sender: TObject);
    procedure Button2Click(Sender: TObject);
    procedure Button3Click(Sender: TObject);
    procedure btnIndexRagFileClick(Sender: TObject);
    procedure chkUseRagClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
  private
    const
      CEmbeddingModelPath = 'D:\embeddinggemma-300m-qat-Q8_0.gguf';
      CRagDbFileName = 'rag.sqlite';
      // Conservative Vulkan profile for the MX450, where the chat and
      // embedding models share a small VRAM budget.
      CInferenceMaxContext = 1024;
      CEmbeddingMaxContext = 128;
      CRagChunkWords = 64;
      CRagOverlapWords = 8;
    var
      LSession: TVdxSession;
      FWorker: TThread;
      FCancel: Integer;
      FRagDbPath: string;
      FHasExistingRag: Boolean;
    procedure GenerationFinished(Sender: TObject);
    procedure ReleaseWorker;
    procedure IndexRagFile;
    procedure UpdateRagAvailability;
  public
    destructor Destroy; override;
  end;

var
  Form1: TForm1;

implementation

{$R *.dfm}

procedure TForm1.FormCreate(Sender: TObject);
begin
  FRagDbPath := TPath.Combine(ExtractFilePath(ParamStr(0)), CRagDbFileName);
  OpenDialog1.Filter := 'Arquivos de texto|*.txt;*.md;*.csv;*.json;*.log|' +
    'Todos os arquivos|*.*';
  UpdateRagAvailability;
end;

procedure TForm1.UpdateRagAvailability;
begin
  FHasExistingRag := TFile.Exists(FRagDbPath);
  chkLoadExistingRag.Enabled := chkUseRag.Checked and FHasExistingRag;
  chkLoadExistingRag.Checked := FHasExistingRag;
  if FHasExistingRag then
    chkLoadExistingRag.Caption := 'Carregar RAG anterior'
  else
    chkLoadExistingRag.Caption := 'Nenhum RAG anterior encontrado';
end;

procedure TForm1.ReleaseWorker;
begin
  if FWorker = nil then
    Exit;
  TInterlocked.Exchange(FCancel, 1);
  FWorker.WaitFor;
  TThread.RemoveQueuedEvents(FWorker);
  FreeAndNil(FWorker);
end;

destructor TForm1.Destroy;
begin
  ReleaseWorker;
  FreeAndNil(LSession);
  inherited;
end;

procedure TForm1.Button1Click(Sender: TObject);
var
  RetrievalConfig: TVdxRetrievalConfig;
  EmbedderPath: string;
  MemoryDbPath: string;
begin
  ReleaseWorker;
  FreeAndNil(LSession);

  EmbedderPath := '';
  MemoryDbPath := '';
  if chkUseRag.Checked then
  begin
    if not TFile.Exists(CEmbeddingModelPath) then
    begin
      Memo1.Lines.Add('Modelo de embeddings não encontrado: ' +
        CEmbeddingModelPath);
      Exit;
    end;
    EmbedderPath := CEmbeddingModelPath;
    MemoryDbPath := FRagDbPath;
  end;

  ForceDirectories(ExtractFilePath(FRagDbPath));
  LSession := TVdxSession.Create;
  LSession.SetStatusCallback(
    procedure(const S: string; const U: Pointer)
    begin
      TThread.Queue(nil,
        procedure
        begin
          if not (csDestroying in ComponentState) then
            Memo1.Lines.Add(S);
        end);
    end, nil);
  LSession.SetRagProgressCallback(
    procedure(const ACompleted, ATotal: Integer)
    begin
      TThread.Queue(nil,
        procedure
        begin
          if not (csDestroying in ComponentState) then
          begin
            if ATotal > 0 then
              ProgressBar1.Max := ATotal
            else
              ProgressBar1.Max := 1;
            ProgressBar1.Position := ACompleted;
          end;
        end);
    end);
  LSession.SetTokenCallback(
    procedure(const S: string; const U: Pointer)
    begin
      TThread.Queue(nil,
        procedure
        begin
          Memo2.SelStart := Length(Memo2.Text);
          Memo2.SelLength := 0;
          Memo2.SelText := S;
        end);
    end, nil);

  LSession.SetCancelCallback(
    function(const U: Pointer): Boolean
    begin
      Result := TInterlocked.CompareExchange(FCancel, 0, 0) <> 0;
    end, nil);

  Button3.Enabled := False;
  try
    if not LSession.LoadModel(Trim(Edit1.Text), MemoryDbPath,
      EmbedderPath, CInferenceMaxContext, -1, CEmbeddingMaxContext) then
    begin
      Memo1.Lines.Add(LSession.GetErrors.ToString);
      FreeAndNil(LSession);
      Exit;
    end;

    // The selected chat model changes generation only. When enabled, RAG
    // always uses the fixed EmbeddingGemma model above.
    if chkUseRag.Checked and not LSession.IsRagReady then
    begin
      Memo1.Lines.Add('O RAG não pôde ser inicializado:' + sLineBreak +
        LSession.GetErrors.ToString);
      FreeAndNil(LSession);
      Exit;
    end;

    RetrievalConfig := TVdxSession.DefaultRetrievalConfig;
    RetrievalConfig.Enabled := chkUseRag.Checked;
    RetrievalConfig.TopK := 3;
    RetrievalConfig.MinScore := -1.0;
    LSession.SetRetrievalConfig(RetrievalConfig);
    LSession.SetContextOnlyPrompt(chkUseRag.Checked);

    if chkUseRag.Checked and FHasExistingRag and
       not chkLoadExistingRag.Checked then
    begin
      LSession.ClearHistory;
      Memo1.Lines.Add('RAG anterior removido; um novo RAG será criado.');
    end
    else if chkUseRag.Checked and FHasExistingRag then
      Memo1.Lines.Add('RAG anterior carregado: ' + FRagDbPath)
    else if chkUseRag.Checked then
      Memo1.Lines.Add('RAG novo será salvo em: ' + FRagDbPath);

    Button3.Enabled := True;
    if chkUseRag.Checked and (Trim(edRagFile.Text) <> '') then
      IndexRagFile;
  except
    on E: Exception do
    begin
      Memo1.Lines.Add(E.Message);
      FreeAndNil(LSession);
    end;
  end;
end;

procedure TForm1.Button2Click(Sender: TObject);
begin
  ReleaseWorker;
  FreeAndNil(LSession);
  Button3.Enabled := False;
  Memo1.Lines.Add('Modelo descarregado.');
end;

procedure TForm1.GenerationFinished(Sender: TObject);
begin
  Button1.Enabled := True;
  Button2.Enabled := True;
  Button3.Enabled := LSession <> nil;
  Edit1.Enabled := True;
  edRagFile.Enabled := True;
  btnIndexRagFile.Enabled := True;
  chkUseRag.Enabled := True;
  chkLoadExistingRag.Enabled := chkUseRag.Checked and FHasExistingRag;
  if (LSession <> nil) and LSession.GetErrors.HasErrors then
    Memo1.Lines.Add(LSession.GetErrors.ToString);
end;

procedure TForm1.IndexRagFile;
var
  FileName: string;
  Text: string;
begin
  if not chkUseRag.Checked then
  begin
    Memo1.Lines.Add('Ative "Usar RAG" e recarregue o modelo para indexar arquivos.');
    Exit;
  end;
  FileName := Trim(edRagFile.Text);
  if FileName = '' then
    Exit;
  if LSession = nil then
  begin
    Memo1.Lines.Add('Carregue o modelo antes de indexar o arquivo.');
    Exit;
  end;
  if FWorker <> nil then
  begin
    Memo1.Lines.Add('Aguarde a operação atual terminar.');
    Exit;
  end;
  if not TFile.Exists(FileName) then
  begin
    Memo1.Lines.Add('Arquivo para RAG não encontrado: ' + FileName);
    Exit;
  end;

  try
    Text := TFile.ReadAllText(FileName);
    if Text.Trim.IsEmpty then
    begin
      Memo1.Lines.Add('O arquivo para RAG está vazio: ' + FileName);
      Exit;
    end;

    ProgressBar1.Max := 1;
    ProgressBar1.Position := 0;
    Button1.Enabled := False;
    Button2.Enabled := False;
    Button3.Enabled := False;
    Edit1.Enabled := False;
    edRagFile.Enabled := False;
    btnIndexRagFile.Enabled := False;
    chkUseRag.Enabled := False;
    chkLoadExistingRag.Enabled := False;

    // Indexing is intentionally off the UI thread. The RAG progress callback
    // queues each completed chunk back to ProgressBar1.
    FWorker := TThread.CreateAnonymousThread(
      procedure
      var
        DocumentId: Int64;
        ErrorText: string;
      begin
        try
          // AddDocument replaces chunks whose source is this same file, so a
          // changed file can be re-indexed without duplicate RAG context.
          DocumentId := LSession.AddDocument(FileName, ExtractFileName(FileName),
            Text, CRagChunkWords, CRagOverlapWords, True);
          if DocumentId > 0 then
            TThread.Queue(nil,
              procedure
              begin
                Memo1.Lines.Add(
                  'Arquivo indexado no RAG (substituído se já existia): ' +
                  FileName);
                UpdateRagAvailability;
              end)
          else
          begin
            ErrorText := LSession.GetErrors.ToString;
            TThread.Queue(nil,
              procedure
              begin
                Memo1.Lines.Add('Falha ao indexar o arquivo:' + sLineBreak +
                  ErrorText);
              end);
          end;
        except
          on E: Exception do
          begin
            ErrorText := E.Message;
            TThread.Queue(nil,
              procedure
              begin
                Memo1.Lines.Add('Falha ao ler/indexar o arquivo: ' + ErrorText);
              end);
          end;
        end;
      end);
    FWorker.FreeOnTerminate := False;
    FWorker.OnTerminate := GenerationFinished;
    FWorker.Start;
  except
    on E: Exception do
      Memo1.Lines.Add('Falha ao ler/indexar o arquivo: ' + E.Message);
  end;
end;

procedure TForm1.btnIndexRagFileClick(Sender: TObject);
begin
  if not OpenDialog1.Execute then
    Exit;

  edRagFile.Text := OpenDialog1.FileName;
  IndexRagFile;
end;

procedure TForm1.chkUseRagClick(Sender: TObject);
begin
  UpdateRagAvailability;
  if LSession <> nil then
    Memo1.Lines.Add('A alteração de RAG será aplicada ao recarregar o modelo.');
end;

procedure TForm1.Button3Click(Sender: TObject);
var
  Config: TVdxSamplerConfig;
  UserInput: string;
begin
  if LSession = nil then
  begin
    Memo1.Lines.Add('Carregue o modelo e o RAG primeiro.');
    Exit;
  end;

  ReleaseWorker;
  UserInput := Trim(mPrompt.Text);
  if UserInput = '' then
    Exit;

  Config := TVdxSampler.DefaultConfig;
  Config.Temperature := StrToFloatDef(edTemperatura.Text, 0);
  Config.TopK := StrToIntDef(edTopK.Text, 0);
  Config.TopP := StrToFloatDef(edTopP.Text, 0);
  Config.MinP := StrToFloatDef(edMinP.Text, 0);
  Config.RepeatPenalty := StrToFloatDef(edRepeatPenalty.Text, 0);
  LSession.SetSamplerConfig(Config);

  TInterlocked.Exchange(FCancel, 0);
  Button1.Enabled := False;
  Button2.Enabled := False;
  Button3.Enabled := False;
  Edit1.Enabled := False;
  edRagFile.Enabled := False;
  btnIndexRagFile.Enabled := False;
  chkUseRag.Enabled := False;
  chkLoadExistingRag.Enabled := False;

  Memo2.Lines.Add('');
  Memo2.Lines.Add(UserInput);

  FWorker := TThread.CreateAnonymousThread(
    procedure
    var
      ErrorText: string;
    begin
      try
        LSession.Chat(UserInput, StrToIntDef(edMaxTokens.Text, 256));
      except
        on E: Exception do
        begin
          ErrorText := E.Message;
          TThread.Queue(nil,
            procedure
            begin
              Memo1.Lines.Add(ErrorText);
            end);
        end;
      end;
    end);

  FWorker.FreeOnTerminate := False;
  FWorker.OnTerminate := GenerationFinished;
  FWorker.Start;
end;

end.
