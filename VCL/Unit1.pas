unit Unit1;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants,
  System.Classes, System.SyncObjs, Vcl.Graphics, Vcl.Controls, Vcl.Forms,
  Vcl.Dialogs, Vcl.StdCtrls, VindexLLM.Inference, VindexLLM.Sampler;

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
    procedure Button1Click(Sender: TObject);
    procedure Button2Click(Sender: TObject);
    procedure Button3Click(Sender: TObject);
  private
    LInference: TVdxInference;
    FWorker: TThread;
    FCancel: Integer;
    procedure GenerationFinished(Sender: TObject);
    procedure ReleaseWorker;
  public
    destructor Destroy; override;
  end;

var Form1: TForm1;

implementation

{$R *.dfm}

procedure TForm1.ReleaseWorker;
begin
  if FWorker=nil then Exit;
  TInterlocked.Exchange(FCancel,1);
  FWorker.WaitFor;
  TThread.RemoveQueuedEvents(FWorker);
  FreeAndNil(FWorker);
end;

destructor TForm1.Destroy;
begin
  ReleaseWorker;
  FreeAndNil(LInference);
  inherited;
end;

procedure TForm1.Button1Click(Sender: TObject);
begin
  ReleaseWorker;
  FreeAndNil(LInference);
  LInference := TVdxInference.Create;

  LInference.SetStatusCallback(
    procedure(const S: string; const U: Pointer)
    begin Memo1.Lines.Add(S); end);

  LInference.SetTokenCallback(
    procedure(const S: string; const U: Pointer)
    begin
      TThread.Queue(FWorker,
        procedure
        begin
          Memo2.SelStart := Length(Memo1.Text);
          Memo2.SelLength := 0;
          Memo2.SelText := S;
        end);
    end,nil);

  LInference.SetCancelCallback(
    function(const U: Pointer): Boolean
    begin Result := TInterlocked.CompareExchange(FCancel,0,0)<>0; end,nil);

  Button3.Enabled := False;
  try
    if not LInference.LoadModel(Trim(Edit1.Text),4096) then
    begin
      Memo1.Lines.Add(LInference.GetErrors.ToString);
      FreeAndNil(LInference);
      Exit;
    end;
    Button3.Enabled := True;
  except
    on E: Exception do
    begin
      Memo1.Lines.Add(E.Message);
      FreeAndNil(LInference);
    end;
  end;
end;

procedure TForm1.Button2Click(Sender: TObject);
begin
  ReleaseWorker;
  FreeAndNil(LInference);
  Button3.Enabled := False;
  Memo1.Lines.Add('Modelo descarregado.');
end;

procedure TForm1.GenerationFinished(Sender: TObject);
begin
  Button1.Enabled := True;
  Button2.Enabled := True;
  Button3.Enabled := LInference<>nil;
  Edit1.Enabled := True;
  if (LInference<>nil) and LInference.GetErrors.HasErrors then
    Memo1.Lines.Add(LInference.GetErrors.ToString);
end;

procedure TForm1.Button3Click(Sender: TObject);
var Config: TVdxSamplerConfig; Prompt: string;
begin
  if LInference=nil then
  begin Memo1.Lines.Add('Carregue o modelo primeiro.'); Exit; end;

  ReleaseWorker;
  Prompt := mPrompt.Text;

  if Trim(Prompt)='' then Exit;

  Config := TVdxSampler.DefaultConfig;
  Config.Temperature := StrToFloatDef(edTemperatura.Text, 0);
  Config.TopK := StrToIntDef(edTopK.Text, 0);
  Config.TopP := StrToFloatDef(edTopP.Text, 0);
  Config.MinP := StrToFloatDef(edMinP.Text, 0);
  Config.RepeatPenalty := StrToFloatDef(edRepeatPenalty.Text, 0);
  LInference.SetSamplerConfig(Config);
  // Each click starts an independent prompt. The SSM state resets at position zero.
  LInference.ResetKVCache;
  TInterlocked.Exchange(FCancel,0);
  Button1.Enabled := False;
  Button2.Enabled := False;
  Button3.Enabled := False;
  Edit1.Enabled := False;

  Memo2.Lines.Add('');
  Memo2.Lines.Add(Prompt);

  Prompt := '"mensagem do cliente: ' + Prompt + '"';

  Prompt := Prompt + #13#10 + 'Classifique nesse formato: {"resposta": <resposta da loja>, "mensagem_cliente":<repita a mensagem>, "classificacao_mensagem_cliente": <CATEGORIA>}';
  Prompt := Prompt + #13#10 + 'Classificações disponíveis: [cardapio, promocao, atendente, status-pedido, ';
  Prompt := Prompt + 'tempo-preparo-entrega, metodo-pagamento, endereco-loja, horario-funcionamento, ';
  Prompt := Prompt + 'fazer-pedido, reclamacao, elogio, saudacao, outros]';

  FWorker := TThread.CreateAnonymousThread(
    procedure
    var ErrorText: string;
    begin
      try
        LInference.Generate(Prompt,StrToIntDef(edMaxTokens.Text, 256));
      except
        on E: Exception do
        begin
          ErrorText := E.Message;
          TThread.Queue(TThread.CurrentThread,
            procedure begin Memo1.Lines.Add(ErrorText); end);
        end;
      end;
    end);

  FWorker.FreeOnTerminate := False;
  FWorker.OnTerminate := GenerationFinished;
  FWorker.Start;
end;

end.
