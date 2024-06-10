# Projeto: Identificação de Ferramentas Cirúrgicas usando YOLO

## Download do dataSet: 
https://drive.google.com/drive/u/2/folders/1RA81Lg43XjnsvPSojFPNcYORbS3pb1Re

#### Descrição
Este projeto é parte do trabalho final para a matéria de Inteligência Artificial na Universidade de Passo Fundo, cursada no 5º semestre. O objetivo do projeto é desenvolver um sistema de identificação de ferramentas cirúrgicas utilizando a técnica de detecção de objetos YOLO (You Only Look Once). O sistema será capaz de reconhecer e classificar diferentes ferramentas cirúrgicas em imagens, contribuindo para a automação e melhoria da precisão em ambientes médicos.

### Instalação
1. Ter python 3.8 ou mais instalado na maquina:
https://www.python.org/downloads/

2. Confira se o pip está instalado:
2.2 no terminal escreva
```bash
pip --version
```

Caso ainda não esteja instalado, instalar usando o comando:
```bash
python -m ensurepip --default-pip
```

3. Intalar ultralytics com pip
```bash
 pip install ultralytics
```


#### Autores
Lucas Friedrich - 168238
Leonardo Salient - 179770

### Estrutura do Projeto
- data: Contém os conjuntos de dados utilizados para treinamento e validação. 
- images: Imagens de ferramentas cirúrgicas.
- annotations: Anotações das imagens no formato YOLO.
- models: Modelos treinados e checkpoints.
- scripts: Scripts para pré-processamento de dados, treinamento e avaliação do modelo.
- results: Resultados e métricas de avaliação do modelo.

### Contribuição
#### Se você deseja contribuir com o projeto, por favor, siga os seguintes passos:

1. Faça um fork do repositório.
2. Crie uma branch para sua feature (git checkout -b feature/sua-feature).
3. Commit suas mudanças (git commit -m 'Adiciona nova feature').
4. Faça um push para a branch (git push origin feature/sua-feature).
5. Abra um Pull Request.
