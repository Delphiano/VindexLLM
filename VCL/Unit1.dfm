object Form1: TForm1
  Left = 0
  Top = 0
  Caption = 'VindexLLM RAG'
  ClientHeight = 777
  ClientWidth = 1118
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -12
  Font.Name = 'Segoe UI'
  Font.Style = []
  OnCreate = FormCreate
  TextHeight = 15
  object Label1: TLabel
    Left = 23
    Top = 59
    Width = 66
    Height = 15
    Caption = 'Temperatura'
  end
  object Label2: TLabel
    Left = 239
    Top = 59
    Width = 31
    Height = 15
    Caption = 'Top_K'
  end
  object Label3: TLabel
    Left = 23
    Top = 99
    Width = 31
    Height = 15
    Caption = 'Top_P'
  end
  object Label4: TLabel
    Left = 215
    Top = 99
    Width = 33
    Height = 15
    Caption = 'Min_P'
  end
  object Label5: TLabel
    Left = 31
    Top = 139
    Width = 80
    Height = 15
    Caption = 'Repeat_Penalty'
  end
  object Label6: TLabel
    Left = 263
    Top = 139
    Width = 63
    Height = 15
    Caption = 'Max_tokens'
  end
  object Label7: TLabel
    Left = 23
    Top = 182
    Width = 89
    Height = 15
    Caption = 'Arquivo para RAG'
  end
  object Button1: TButton
    Left = 479
    Top = 16
    Width = 75
    Height = 25
    Caption = 'Carregar'
    TabOrder = 0
    OnClick = Button1Click
  end
  object Memo1: TMemo
    Left = 560
    Top = 17
    Width = 531
    Height = 696
    ScrollBars = ssVertical
    TabOrder = 1
  end
  object Button2: TButton
    Left = 1016
    Top = 728
    Width = 75
    Height = 25
    Caption = 'Descarregar'
    TabOrder = 2
    OnClick = Button2Click
  end
  object Button3: TButton
    Left = 398
    Top = 411
    Width = 75
    Height = 25
    Caption = 'Gerar'
    Enabled = False
    TabOrder = 3
    OnClick = Button3Click
  end
  object ProgressBar1: TProgressBar
    Left = 23
    Top = 411
    Width = 359
    Height = 23
    TabOrder = 17
  end
  object mPrompt: TMemo
    Left = 23
    Top = 280
    Width = 450
    Height = 125
    Lines.Strings = (
      'Estava uma maravilha')
    TabOrder = 4
  end
  object Edit1: TEdit
    Left = 23
    Top = 19
    Width = 450
    Height = 23
    TabOrder = 5
    Text = 'D:\Qwen3.5-0.8B-v2.Q4_K_M.gguf'
  end
  object edTemperatura: TEdit
    Left = 96
    Top = 51
    Width = 121
    Height = 23
    TabOrder = 6
    Text = '0.1'
  end
  object edTopK: TEdit
    Left = 276
    Top = 51
    Width = 121
    Height = 23
    TabOrder = 7
    Text = '20'
  end
  object edTopP: TEdit
    Left = 60
    Top = 91
    Width = 121
    Height = 23
    TabOrder = 8
    Text = '0.8'
  end
  object edMinP: TEdit
    Left = 252
    Top = 91
    Width = 121
    Height = 23
    TabOrder = 9
    Text = '0.1'
  end
  object edRepeatPenalty: TEdit
    Left = 127
    Top = 136
    Width = 121
    Height = 23
    TabOrder = 10
    Text = '1'
  end
  object Memo2: TMemo
    Left = 23
    Top = 453
    Width = 531
    Height = 300
    ScrollBars = ssVertical
    TabOrder = 11
  end
  object edMaxTokens: TEdit
    Left = 332
    Top = 136
    Width = 121
    Height = 23
    TabOrder = 12
    Text = '1000'
  end
  object edRagFile: TEdit
    Left = 23
    Top = 202
    Width = 355
    Height = 23
    TabOrder = 13
  end
  object btnIndexRagFile: TButton
    Left = 384
    Top = 201
    Width = 89
    Height = 25
    Caption = 'Indexar arquivo'
    TabOrder = 14
    OnClick = btnIndexRagFileClick
  end
  object chkLoadExistingRag: TCheckBox
    Left = 23
    Top = 249
    Width = 218
    Height = 17
    Caption = 'Nenhum RAG anterior encontrado'
    Enabled = False
    TabOrder = 15
  end
  object chkUseRag: TCheckBox
    Left = 23
    Top = 231
    Width = 97
    Height = 17
    Caption = 'Usar RAG'
    Checked = True
    State = cbChecked
    TabOrder = 16
    OnClick = chkUseRagClick
  end
  object OpenDialog1: TOpenDialog
    Left = 496
    Top = 248
  end
end
