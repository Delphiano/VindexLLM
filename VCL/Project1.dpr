program Project1;

uses
  Vcl.Forms,
  Unit1 in 'Unit1.pas' {Form1},
  VindexLLM.Model.Gemma3 in '..\src\VindexLLM.Model.Gemma3.pas',
  VindexLLM.Session in '..\src\VindexLLM.Session.pas',
  VindexLLM.Tokenizer in '..\src\VindexLLM.Tokenizer.pas',
  VindexLLM.Inference in '..\src\VindexLLM.Inference.pas',
  VindexLLM.Model.Llama in '..\src\VindexLLM.Model.Llama.pas',
  VindexLLM.Model.Ministral3 in '..\src\VindexLLM.Model.Ministral3.pas',
  VindexLLM.Model.Qwen35 in '..\src\VindexLLM.Model.Qwen35.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TForm1, Form1);
  Application.Run;
end.
