object Form1: TForm1
  Left = 0
  Top = 0
  Caption = 'Form1'
  ClientHeight = 598
  ClientWidth = 581
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -12
  Font.Name = 'Segoe UI'
  Font.Style = []
  TextHeight = 15
  object Button1: TButton
    Left = 479
    Top = 16
    Width = 75
    Height = 25
    Caption = 'Button1'
    TabOrder = 0
    OnClick = Button1Click
  end
  object Memo1: TMemo
    Left = 23
    Top = 360
    Width = 531
    Height = 193
    TabOrder = 1
  end
  object Button2: TButton
    Left = 479
    Top = 565
    Width = 75
    Height = 25
    Caption = 'Button2'
    TabOrder = 2
    OnClick = Button2Click
  end
  object Button3: TButton
    Left = 479
    Top = 47
    Width = 75
    Height = 282
    Caption = 'Button3'
    TabOrder = 3
    OnClick = Button3Click
  end
  object mPrompt: TMemo
    Left = 23
    Top = 48
    Width = 450
    Height = 281
    Lines.Strings = (
      'Mensagem do usu'#225'rio: "Quanto t'#225' a pizza grande de calabresa?"'
      '--- CATEGORIA: FAZER-PEDIDO'
      '  - Quero uma pizza de calabresa'
      '  - Vou querer uma pizza meio a meio para entrega'
      '  - Vou querer uma pizza de frango para entrega'
      '  - Pode anotar uma pizza meio a meio para mim?'
      '  - Quero fazer pedido de jantar'
      ''
      '--- CATEGORIA: CARDAPIO'
      '  - Quanto custa pizza de mu'#231'arela?'
      '  - Quanto custa pizza de pepperoni?'
      '  - Quanto custa pizza de frango?'
      '  - Qual o valor do X-Calabresa?'
      '  - Qual '#233' o tamanho da por'#231#227'o de calabresa acebolada?'
      ''
      '--- CATEGORIA: STATUS-PEDIDO'
      '  - Quanto tempo ainda vai levar?'
      '  - J'#225' t'#225' a caminho de casa?'
      '  - Quanto tempo pro meu pedido sair?'
      '  - J'#225' saiu para entrega?'
      '  - J'#225' saiu pra entrega?'
      ''
      '--- CATEGORIA: TEMPO-ENTREGA'
      '  - Quanto tempo demora a entrega?'
      '  - Quanto tempo pro delivery?'
      '  - Quanto demora uma pizza grande?'
      '  - Quanto leva para montar o lanche?'
      '  - Quanto tempo para assar e enviar?'
      ''
      '--- CATEGORIA: PROMOCAO'
      '  - Qual '#233' a promo'#231#227'o de delivery?'
      '  - Qual '#233' a promo'#231#227'o de sobremesa?'
      '  - Qual '#233' a promo'#231#227'o de combo casal?'
      '  - Qual '#233' a promo'#231#227'o de X-Bacon?'
      '  - Qual '#233' a promo'#231#227'o de milkshake?'
      ''
      '--- CATEGORIA: HORARIO-FUNCIONAMENTO'
      '  - Que horas param de entregar?'
      '  - Qual '#233' o hor'#225'rio de quarta?'
      '  - Qual '#233' o hor'#225'rio de ter'#231'a?'
      '  - Qual o expediente de hoje?'
      '  - Qual o hor'#225'rio de abertura?'
      ''
      '--- CATEGORIA: NOSSO-ENDERECO'
      '  - Qual a rua de voc'#234's?'
      '  - Onde fica o ponto de entrega?'
      '  - Qual o bairro de voc'#234's?'
      '  - Qual o endere'#231'o pra eu buscar?'
      '  - Qual o ponto de refer'#234'ncia?'
      ''
      '--- CATEGORIA: ATENDENTE'
      '  - Quero falar com pessoa real'
      '  - Quero suporte humano para tratar do meu reembolso'
      '  - Tem atendente livre?'
      '  - Quero falar com atendente de verdade'
      '  - Falar com atendente agora'
      ''
      '--- CATEGORIA: METODO-PAGAMENTO'
      '  - Quais as formas de pagamento?'
      '  - Aceita link de pagamento?'
      '  - Aceitam Pix copia e cola?'
      '  - Posso usar Visa no delivery?'
      '  - Posso pagar no ato da entrega?'
      ''
      '--- CATEGORIA: ELOGIO'
      '  - Lanche gigante e delicioso'
      '  - O pizza estava excelente'
      '  - Melhor pizza que j'#225' comi'
      '  - Nunca vi lanche t'#227'o bom'
      '  - Comida caseira deliciosa'
      ''
      '--- CATEGORIA: RECLAMACAO'
      '  - Quero reclamar porque mandaram o sabor errado'
      '  - Quero reclamar porque o lanche chegou amassado'
      '  - Pizza fria e dura'
      '  - Batata dura e fria'
      '  - A entrega foi para outra casa'
      ''
      '--- CATEGORIA: SAUDACAO'
      '  - Boa tarde galera'
      '  - Boa tarde, beleza?'
      '  - Bom dia tudo bem?'
      '  - Boa tarde tudo bom?'
      '  - Bom dia galera'
      ''
      
        'Pegue a mensagem do usu'#225'rio e classfique em uma das categorias i' +
        'nformadas'
      'Classifique nesse formato: {"classificacao": <CATEGORIA>}'
      'Mensagem do usu'#225'rio: "Quanto t'#225' a pizza grande de calabresa?"'
      '--- CATEGORIA: FAZER-PEDIDO'
      '  - Quero uma pizza de calabresa'
      '  - Vou querer uma pizza meio a meio para entrega'
      '  - Vou querer uma pizza de frango para entrega'
      '  - Pode anotar uma pizza meio a meio para mim?'
      '  - Quero fazer pedido de jantar'
      ''
      '--- CATEGORIA: CARDAPIO'
      '  - Quanto custa pizza de mu'#231'arela?'
      '  - Quanto custa pizza de pepperoni?'
      '  - Quanto custa pizza de frango?'
      '  - Qual o valor do X-Calabresa?'
      '  - Qual '#233' o tamanho da por'#231#227'o de calabresa acebolada?'
      ''
      '--- CATEGORIA: STATUS-PEDIDO'
      '  - Quanto tempo ainda vai levar?'
      '  - J'#225' t'#225' a caminho de casa?'
      '  - Quanto tempo pro meu pedido sair?'
      '  - J'#225' saiu para entrega?'
      '  - J'#225' saiu pra entrega?'
      ''
      '--- CATEGORIA: TEMPO-ENTREGA'
      '  - Quanto tempo demora a entrega?'
      '  - Quanto tempo pro delivery?'
      '  - Quanto demora uma pizza grande?'
      '  - Quanto leva para montar o lanche?'
      '  - Quanto tempo para assar e enviar?'
      ''
      '--- CATEGORIA: PROMOCAO'
      '  - Qual '#233' a promo'#231#227'o de delivery?'
      '  - Qual '#233' a promo'#231#227'o de sobremesa?'
      '  - Qual '#233' a promo'#231#227'o de combo casal?'
      '  - Qual '#233' a promo'#231#227'o de X-Bacon?'
      '  - Qual '#233' a promo'#231#227'o de milkshake?'
      ''
      '--- CATEGORIA: HORARIO-FUNCIONAMENTO'
      '  - Que horas param de entregar?'
      '  - Qual '#233' o hor'#225'rio de quarta?'
      '  - Qual '#233' o hor'#225'rio de ter'#231'a?'
      '  - Qual o expediente de hoje?'
      '  - Qual o hor'#225'rio de abertura?'
      ''
      '--- CATEGORIA: NOSSO-ENDERECO'
      '  - Qual a rua de voc'#234's?'
      '  - Onde fica o ponto de entrega?'
      '  - Qual o bairro de voc'#234's?'
      '  - Qual o endere'#231'o pra eu buscar?'
      '  - Qual o ponto de refer'#234'ncia?'
      ''
      '--- CATEGORIA: ATENDENTE'
      '  - Quero falar com pessoa real'
      '  - Quero suporte humano para tratar do meu reembolso'
      '  - Tem atendente livre?'
      '  - Quero falar com atendente de verdade'
      '  - Falar com atendente agora'
      ''
      '--- CATEGORIA: METODO-PAGAMENTO'
      '  - Quais as formas de pagamento?'
      '  - Aceita link de pagamento?'
      '  - Aceitam Pix copia e cola?'
      '  - Posso usar Visa no delivery?'
      '  - Posso pagar no ato da entrega?'
      ''
      '--- CATEGORIA: ELOGIO'
      '  - Lanche gigante e delicioso'
      '  - O pizza estava excelente'
      '  - Melhor pizza que j'#225' comi'
      '  - Nunca vi lanche t'#227'o bom'
      '  - Comida caseira deliciosa'
      ''
      '--- CATEGORIA: RECLAMACAO'
      '  - Quero reclamar porque mandaram o sabor errado'
      '  - Quero reclamar porque o lanche chegou amassado'
      '  - Pizza fria e dura'
      '  - Batata dura e fria'
      '  - A entrega foi para outra casa'
      ''
      '--- CATEGORIA: SAUDACAO'
      '  - Boa tarde galera'
      '  - Boa tarde, beleza?'
      '  - Bom dia tudo bem?'
      '  - Boa tarde tudo bom?'
      '  - Bom dia galera'
      ''
      
        'Pegue a mensagem do usu'#225'rio e classfique em uma das categorias i' +
        'nformadas'
      'Classifique nesse formato: {"classificacao": <CATEGORIA>}')
    TabOrder = 4
  end
  object Edit1: TEdit
    Left = 23
    Top = 19
    Width = 450
    Height = 23
    TabOrder = 5
    Text = 'D:\gemma-3-4b-it-q4_0.gguf'
  end
end
