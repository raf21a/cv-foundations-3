# cv-foundations-3
Third project of discipline Computer Vision Foundations

## Conteúdo
 1. [Requisitos](#requisitos)
 2. [Estrutura](#estrutura)
 3. [Uso](#uso)

## Requisitos 
1.  Python 3.5.2	
2.  OpenCV 3.3.0

## Estrutura
- Pasta relatorio com código fonte do relatório
- Arquivo Araujo_Pedro__Sousa_Rafael.pdf com o relatório
- Pasta src contendo o código do projeto: pd3.py.

## Uso
- A partir do diretório raiz rodar com configurações padrão:
	```bash
	python ./src/pd3.py --r[número do requisito]
	```
-  [número do requisito] corresponde a 1, 2 ou 3 dependendo do requisito a ser testado.
- Pode-se customizar o uso do programa por meio de flags opcionais:
	- --img_l para indicar caminho para imagem da esquerda
	- --img_r para indicar caminho para imagem da direita
	- --help para obter ajuda sobre o uso do programa
- [Repositório do github](https://github.com/raf21a/cv-foundations-3)
- Requisito 1:
	- Se não usar flag para indicar imagens, será usada as imagens da planta
	- Selecionar parâmetros até ficar satisfeito com o resultado, apertar esc.
	- Mapas seram salvos na raiz do projeto
- Requisito 2:
	- Aparecerão as imagens retificadas. Apertar esc.
	- A partir daqui, mesmo procedimento do requisito 1
- Requisito 3:
	- Aparecerão as imagens retificadas. Apertar esc.
	- Selecionar parâmetros até ficar satisfeito com o resultado, apertar esc.
	- Usar o mouse para clicar os pontos cujos quais deseja-se medir a distância.
