program Project1;

uses
  Vcl.Forms,
  Unit1 in 'Unit1.pas' {Form1},
  UTest.Model.Gemma3 in 'UTest.Model.Gemma3.pas',
  VindexLLM.Tokenizer in '..\src\VindexLLM.Tokenizer.pas',
  VindexLLM.Inference in '..\src\VindexLLM.Inference.pas',
  VindexLLM.Model.Llama in '..\src\VindexLLM.Model.Llama.pas',
  VindexLLM.Model.Mistral3 in '..\src\VindexLLM.Model.Mistral3.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TForm1, Form1);
  Application.Run;
end.
