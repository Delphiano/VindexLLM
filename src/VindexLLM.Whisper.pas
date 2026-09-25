unit VindexLLM.Whisper;

interface

uses
  System.SysUtils;

type
  EWhisperError = class(Exception);

  // Small, dependency-free bridge to the official whisper.cpp command-line
  // program.  Keeping this outside TVdxSession is intentional: Whisper has a
  // speech-specific encoder/decoder and is not a text-generation GGUF model.
  TVdxWhisperCli = class
  private
    FExecutablePath: string;
    FModelPath: string;
    FLanguage: string;
    FThreads: Integer;
    function BuildArguments(const AAudioFile, AOutputBase: string): string;
    function Run(const ACommandLine: string; out AOutput: string): Cardinal;
  public
    constructor Create;
    function Transcribe(const AAudioFile: string): string;
    property ExecutablePath: string read FExecutablePath write FExecutablePath;
    property ModelPath: string read FModelPath write FModelPath;
    property Language: string read FLanguage write FLanguage;
    property Threads: Integer read FThreads write FThreads;
  end;

implementation

uses
  Winapi.Windows,
  System.Classes,
  System.IOUtils;

function QuoteArgument(const AValue: string): string;
begin
  // Model and file paths supplied by the UI are passed as one argument.
  Result := '"' + AValue.Replace('"', '\"') + '"';
end;

constructor TVdxWhisperCli.Create;
begin
  inherited Create;
  FExecutablePath := 'whisper-cli.exe';
  FLanguage := 'pt';
  FThreads := 0;
end;

function TVdxWhisperCli.BuildArguments(const AAudioFile,
  AOutputBase: string): string;
begin
  Result := '--model ' + QuoteArgument(FModelPath) +
    ' --file ' + QuoteArgument(AAudioFile) +
    ' --language ' + QuoteArgument(FLanguage) +
    ' --output-txt --output-file ' + QuoteArgument(AOutputBase);
  if FThreads > 0 then
    Result := Result + ' --threads ' + FThreads.ToString;
end;

function TVdxWhisperCli.Run(const ACommandLine: string;
  out AOutput: string): Cardinal;
var
  LSecurity: TSecurityAttributes;
  LStartupInfo: TStartupInfo;
  LProcessInfo: TProcessInformation;
  LReadPipe: THandle;
  LWritePipe: THandle;
  LAvailable: Cardinal;
  LRead: Cardinal;
  LExitCode: Cardinal;
  LBuffer: TBytes;
  LText: TStringBuilder;
  LCommand: string;
  LFinished: Boolean;
begin
  LReadPipe := 0;
  LWritePipe := 0;
  LText := TStringBuilder.Create;
  try
    ZeroMemory(@LSecurity, SizeOf(LSecurity));
    LSecurity.nLength := SizeOf(LSecurity);
    LSecurity.bInheritHandle := True;
    if not CreatePipe(LReadPipe, LWritePipe, @LSecurity, 0) then
      RaiseLastOSError;
    if not SetHandleInformation(LReadPipe, HANDLE_FLAG_INHERIT, 0) then
      RaiseLastOSError;

    ZeroMemory(@LStartupInfo, SizeOf(LStartupInfo));
    LStartupInfo.cb := SizeOf(LStartupInfo);
    LStartupInfo.dwFlags := STARTF_USESHOWWINDOW or STARTF_USESTDHANDLES;
    LStartupInfo.wShowWindow := SW_HIDE;
    LStartupInfo.hStdOutput := LWritePipe;
    LStartupInfo.hStdError := LWritePipe;
    LStartupInfo.hStdInput := GetStdHandle(STD_INPUT_HANDLE);
    ZeroMemory(@LProcessInfo, SizeOf(LProcessInfo));

    LCommand := ACommandLine;
    if not CreateProcess(nil, PChar(LCommand), nil, nil, True,
      CREATE_NO_WINDOW, nil, nil, LStartupInfo, LProcessInfo) then
      RaiseLastOSError;
    try
      CloseHandle(LWritePipe);
      LWritePipe := 0;
      SetLength(LBuffer, 8192);
      repeat
        LFinished := WaitForSingleObject(LProcessInfo.hProcess, 25) = WAIT_OBJECT_0;
        while PeekNamedPipe(LReadPipe, nil, 0, nil, @LAvailable, nil) and
          (LAvailable > 0) do
        begin
          if LAvailable > Cardinal(Length(LBuffer)) then
            SetLength(LBuffer, LAvailable);
          if not ReadFile(LReadPipe, LBuffer[0], Length(LBuffer), LRead, nil) then
            Break;
          if LRead > 0 then
            LText.Append(TEncoding.UTF8.GetString(LBuffer, 0, LRead));
        end;
      until LFinished;
      GetExitCodeProcess(LProcessInfo.hProcess, LExitCode);
      Result := LExitCode;
    finally
      CloseHandle(LProcessInfo.hThread);
      CloseHandle(LProcessInfo.hProcess);
    end;
    AOutput := LText.ToString;
  finally
    if LWritePipe <> 0 then
      CloseHandle(LWritePipe);
    if LReadPipe <> 0 then
      CloseHandle(LReadPipe);
    LText.Free;
  end;
end;

function TVdxWhisperCli.Transcribe(const AAudioFile: string): string;
var
  LOutputBase: string;
  LTextFile: string;
  LConsoleOutput: string;
  LExitCode: Cardinal;
begin
  if not TFile.Exists(AAudioFile) then
    raise EWhisperError.Create('Arquivo de áudio não encontrado: ' + AAudioFile);
  if Trim(FModelPath) = '' then
    raise EWhisperError.Create('Informe o modelo Whisper (GGUF/GGML compatível com o whisper-cli).');
  if not TFile.Exists(FModelPath) then
    raise EWhisperError.Create('Modelo Whisper não encontrado: ' + FModelPath);

  LOutputBase := TPath.Combine(TPath.GetTempPath, TPath.GetRandomFileName);
  LTextFile := LOutputBase + '.txt';
  try
    LExitCode := Run(QuoteArgument(FExecutablePath) + ' ' +
      BuildArguments(AAudioFile, LOutputBase), LConsoleOutput);
    if LExitCode <> 0 then
      raise EWhisperError.CreateFmt('whisper-cli encerrou com código %d.%s%s',
        [LExitCode, sLineBreak, Trim(LConsoleOutput)]);
    if not TFile.Exists(LTextFile) then
      raise EWhisperError.Create('O whisper-cli não gerou o arquivo de transcrição.' +
        sLineBreak + Trim(LConsoleOutput));
    Result := Trim(TFile.ReadAllText(LTextFile, TEncoding.UTF8));
    if Result = '' then
      raise EWhisperError.Create('A transcrição retornou vazia.');
  finally
    if TFile.Exists(LTextFile) then
      TFile.Delete(LTextFile);
    // Some versions also emit timestamps/json only when those options are
    // requested; no user audio or model file is ever altered here.
  end;
end;

end.
