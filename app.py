### FALTA:
### Transformar o Bairro em Texto com combobox (X)
### Consertar o Slider (X)
### Acrescentar mais features
### Colocar o CEP em numero e texto
### Refinar o Modelo
### Tunning e medidas comparativas
### Hospedar o App no Streamlit Web

import pandas as pd

import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# função para carregar o dataset
@st.cache
def get_data():
    return pd.read_csv("data_real.csv")

# função para treinar o modelo
def train_model():
    data = get_data()    
    x = data.drop(["ALVO", "Bairro", "CEP", "Endereco"],axis=1)    
    y = data["ALVO"]
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(x, y)
    return rf_regressor

# criando um dataframe
data = get_data()

# criando a interface no Streamlit

# título
st.title("Aplicativo - Prevendo Valores de Imóveis")

# subtítulo
st.markdown("Este é um Aplicativo utilizado para exibir a solução de Machine Learning para o problema de predição de valores de imóveis do Rio de Janeiro.")

# verificando o dataset
st.subheader("Segue uma pequena amostra de Imóveis reais, que foram retirados de portais de vendas, para validar a nossa predição:")

# atributos para serem exibidos por padrão
defaultcols = ["Quartos","Area_Total","Bairro","CEP", "Endereco", "ALVO"]

# defindo atributos a partir do multiselect
cols = st.multiselect("Atributos", defaultcols, default=defaultcols)

# exibindo os top 10 registro do dataframe
st.dataframe(data[cols].head(20))

st.subheader("Distribuição de imóveis por preço")

# definindo a faixa de valores
maxValueToken_set = data['ALVO'].max()
minValueToken_set = data['ALVO'].min()
faixa_valores = st.slider("Faixa de preço", minValueToken_set, maxValueToken_set, value=maxValueToken_set, step=1000.)
#print('Faixa: ' + str(faixa_valores))

# filtrando os dados
#dados = data[data['ALVO'].between(left=faixa_valores[0],right=faixa_valores[1])]
dados = data[data['ALVO'].between(left=0,right=faixa_valores)]

# plot a distribuição dos dados
f = px.histogram(dados, x="ALVO", nbins=100, title="Distribuição de Preços")
f.update_xaxes(title="ALVO")
f.update_yaxes(title="Total Imóveis")
st.plotly_chart(f)

st.sidebar.subheader("Defina os atributos do imóvel para predição")

# mapeando dados do usuário para cada atributo
quartos = st.sidebar.number_input("Numero de Quartos", value=data.Quartos.mean())
area_total = st.sidebar.number_input("Área total do imóvel ", value=data.Area_Total.mean())
#bairro = st.sidebar.number_input("Bairro ", value=8)

# inserindo os bairros que estão no sample (copiar do bairrosdesc.csv na pasta da aplicacao)
BAIRROS = {0: "Barra da Tijuca", 1 : "Botafogo", 3: "Copacabana", 4: "Flamengo", 5: "Freguesia(Jacarepaguá)", 6: "Gávea", 7: "Humaitá", 8: "Ipanema", 9: "Jacarepaguá", 10: "Jardim Botânico", 11: "Lagoa", 12: "Laranjeiras",  13: "Leblon", 14: "Leme", 16: "Recreio dos Bandeirantes", 18: "Tijuca", 19:"Vargem Grande"}

#retorna a descricao do bairro pelo codigo
def desc_bairro(bairro):
    return BAIRROS[bairro]

# cria uma caixa de selecao com os bairros para o ususario escolher um
bairro = st.sidebar.selectbox("Selecione o Bairro", options=list(BAIRROS.keys()), format_func=desc_bairro)
st.sidebar.write(f"Você selecionou a opção {bairro} chamada {desc_bairro(bairro)}")
#st.sidebar.selectbox("Selecione o CEP")

# inserindo os CEPs que estão no sample (copiar do CEPdesc.csv na pasta da aplicacao)
CEPS = {
5  :"20550-030 - Rua Marquês de Valença",
6  :"22010-050 - Rua General Ribeiro da Costa",
8  :"22030-040 - Rua Marechal Mascarenhas de Morais",
13 :"22061-030 - Avenida Henrique Dodsworth",
17 :"22081-041 - Avenida Rainha Elizabeth da Bélgica",
18 :"22081-042 - Avenida Rainha Elizabeth da Bélgica",
19 :"22210-080 - Rua Paissandu",
21 :"22220-060 - Rua Machado de Assis",
22 :"22221-080 - Rua Marquesa de Santos",
23 :"22221-140 - Rua Pereira da Silva",
24 :"22230-060 - Rua Marquês de Abrantes",
25 :"22230-061 - Rua Marquês de Abrantes",
26 :"22240-000 - Rua das Laranjeiras",
28 :"22250-020 - Avenida Rui Barbosa",
29 :"22251-030 - Rua Assunção",
31 :"22260-004 - Rua São Clemente",
33 :"22260-009 - Rua São Clemente",
34 :"22260-040 - Rua Ministro Raul Fernandes",
36 :"22261-000 - Rua do Humaitá",
37 :"22270-010 - Rua Voluntários da Pátria",
38 :"22271-100 - Rua Mena Barreto",
39 :"22280-020 - Rua Dona Mariana",
41 :"22281-033 - Rua Real Grandeza",
42 :"22281-034 - Rua Real Grandeza",
43 :"22290-031 - Rua da Passagem",
44 :"22290-070 - Avenida Lauro Sodré",
46 :"22410-001 - Rua Visconde de Pirajá",
47 :"22410-002 - Rua Visconde de Pirajá",
48 :"22411-001 - Rua Barão da Torre",
49 :"22411-010 - Rua Vinícius de Moraes",
51 :"22411-040 - Rua Almirante Saddock de Sá",
52 :"22411-072 - Avenida Epitácio Pessoa",
54 :"22420-040 - Rua Prudente de Morais",
55 :"22420-041 - Rua Prudente de Morais",
56 :"22421-000 - Rua Barão de Jaguaripe",
57 :"22421-030 - Rua Redentor",
60 :"22431-030 - Rua Desembargador Alfredo Russel",
62 :"22440-000 - Rua Almirante Guilhem",
65 :"22441-040 - Rua Adalberto Ferreira",
66 :"22450-030 - Rua Professor Azevedo Marques",
67 :"22450-130 - Rua Timóteo da Costa",
68 :"22450-140 - Rua Sambaíba",
69 :"22450-200 - Rua Igarapava",
70 :"22451-041 - Rua Marquês de São Vicente",
71 :"22451-042 - Rua Marquês de São Vicente",
72 :"22451-045 - Rua Marquês de São Vicente",
74 :"22460-012 - Rua Lópes Quintas",
75 :"22461-000 - Rua Jardim Botânico",
76 :"22461-010 - Rua Visconde da Graça",
77 :"22461-130 - Rua J. Carlos",
78 :"22461-152 - Rua Maria Angélica",
79 :"22461-240 - Rua Pio Correia",
82 :"22470-050 - Rua Jardim Botânico",
83 :"22470-220 - Avenida Alexandre Ferreira",
85 :"22471-004 - Avenida Epitácio Pessoa",
87 :"22471-160 - Rua Ildefonso Simões Lópes",
88 :"22471-180 - Rua Sacopa",
89 :"22471-210 - Rua Fonte da Saudade",
91 :"22620-171 - Avenida Pepe",
92 :"22620-172 - Avenida Lúcio Costa",
93 :"22620-300 - Rua de Paranhos Antunes",
95 :"22621-060 - Avenida Monsenhor Ascaneo",
96 :"22621-200 - Avenida Olegário Maciel",
97 :"22630-011 - Avenida Lúcio Costa",
100:"22743-670 - Rua Geminiano Gois",
101:"22745-200 - Estrada do Guanumbi",
102:"22750-012 - Estrada do Bananal",
103:"22755-001 - Estrada dos Três Rios",
104:"22775-033 - Avenida Vice-Presidente José Alencar",
105:"22775-040 - Avenida Embaixador Abelardo Bueno",
106:"22775-046 - Rua Mário Agostinelli",
107:"22775-057 - Avenida João Cabral de Mello Neto",
108:"22776-000 - Avenida das Acácias da Península",
109:"22776-040 - Rua das Bromélias da Península",
110:"22776-050 - Rua Jacarandás da Península",
111:"22776-070 - Avenida Flamboyants da Península",
112:"22776-090 - Rua Bauhíneas da Península",
114:"22785-595 - Rua Mário Lisboa de Carvalho",
118:"22790-704 - Avenida das Américas",
122:"22793-081 - Avenida das Américas",
123:"22793-082 - Avenida das Américas",
125:"22793-283 - Rua Semi Alzuguir",
126:"22793-295 - Avenida Malibu",
128:"22793-319 - Avenida Rosauro Estellita"}

#retorna a descricao do bairro pelo codigo
def desc_cep(cep):
    return CEPS[cep]

# cria uma caixa de selecao com os bairros para o usuario escolher um
cep = st.sidebar.selectbox("Selecione o CEP", options=list(CEPS.keys()), format_func=desc_cep)
#st.sidebar.write(f"Você selecionou a opção {cep} chamada {desc_cep(cep)}")

# treinando o modelo e deixando ele pronto para responder ao clique do botao
model = train_model()

# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Predição")

# verifica se o botão foi acionado
if btn_predict:
    # realiza a predicao
    result = model.predict([[quartos,area_total,bairro,cep]])
    st.subheader("O valor previsto para o imóvel é:")
    # formata o resultado
    result = "R$ "+str(round(result[0]*10,2))
    # imprime o resultado na tela
    st.write(result)

