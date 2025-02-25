import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dash_bootstrap_templates import ThemeSwitchAIO
from app import *
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image
import nltk
from nltk.corpus import stopwords
from collections import Counter
import flask_caching
from flask_caching import Cache



# Especifique o caminho para a pasta do seu projeto atual
nltk.data.path.append('/home/black_d/Downloads/Dash_Bourds_Python/New_Project_dashboard/data/nltk_data')
stop_words = set(stopwords.words('portuguese'))

'''=============================== Carregar os dados #==============================='''
# Configuração do cache para armazenar os dados
cache = Cache(app.server, config={"CACHE_TYPE": "simple"})

@cache.memoize()
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'hospital_ptbr.csv')
    return pd.read_csv(file_path)

df = load_data()

''' ============================# config_style #================================'''
tab_card = {'height':'100%', 
            'background-color':'#F8BBD0', 
            'border-radius': '15px'
            }

main_config = {
    "margin": {"l": 10, "r": 10, "t": 10, "b": 10}, 
    "plot_bgcolor": "rgba(0,0,0,0)",  # Transparência no fundo do gráfico
    "paper_bgcolor": "rgba(0,0,0,0)",  # Transparência no fundo da área do gráfico
    "font": {"color": "white", "size": 15},
}

config_graph={"displayModeBar": False, "showTips": False}

''' ============================# layout #================================'''

content = html.Div(id="page-content")
app.layout = html.Div([
    dcc.Location(id = 'url', refresh = True), content
])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])

def render_page_content(pathname):
    if pathname == '/':
        return html.Div([
            html.H2('Análise de Sentimentos', 
                    style={
                        'color': 'white', 'background-color':'#F8BBD0', 
                        'textAlign':'center', 'border-radius': '15px'
                        }),
            dbc.Container(children=[
                # Row 1
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row(
                                    dbc.Col(
                                        html.H4('Wordcloud - Feedbacks Positivos', style = {'textAlign':'center'})
                                    )
                                ), 
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id = 'graph1', className = 'dbc', config = config_graph), 
                                        dcc.Interval(id='dummy-input', interval=1*1000, n_intervals=0)
                                    ], sm = 12, md = 12)
                                ], justify = 'center')
                            ])
                        ], style = tab_card)
                    ], sm = 12, lg = 6), 
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row(
                                    dbc.Col(
                                        html.H4('Wordcloud - Feedbacks Negativos', 
                                                style = {'textAlign':'center'})
                                    )
                                ),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id = 'graph2', className = 'dbc', config = config_graph)
                                    ], sm = 12, md = 12)
                                ], justify = 'center')
                            ])
                        ], style = tab_card)
                    ], sm = 12, lg = 6)
                ], className =' g-3 my-auto', style = {'margin-top':'3px'}), 

                # Row 2
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row(
                                    dbc.Col(
                                        html.H4('Distribuição das Avaliações', style = {'textAlign':'center'})
                                    )
                                ), 
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id = 'graph3', className = 'dbc', config = config_graph)
                                    ], sm = 12, md = 12)
                                ], justify = 'center')
                            ])
                        ], style = tab_card)
                    ], sm = 12, lg = 4), 
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row(
                                    dbc.Col(
                                        html.H4('Distribuição dos Sentimentos', 
                                                style = {'textAlign':'center'})
                                    )
                                ),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id = 'graph4', className = 'dbc', config = config_graph)
                                    ], sm = 12, md = 12)
                                ], justify = 'center')
                            ])
                        ], style = tab_card)
                    ], sm = 12, lg = 4), 
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row(
                                    dbc.Col(
                                        html.H4('Distribuição das Avaliações por Sentimeento', 
                                                style = {'textAlign':'center'})
                                    )
                                ),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id = 'graph5', className = 'dbc', config = config_graph)
                                    ], sm = 12, md = 12)
                                ], justify = 'center')
                            ])
                        ], style = tab_card)
                    ], sm = 12, lg = 4), 
                ], className =' g-3 my-auto', style = {'margin-top':'3px'}),

                # Row 3
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        dbc.Row(
                                            dbc.Col(
                                                html.H4('Frequência de Palavras nos Feedbacks Positivas', style = {'textAlign':'center'})
                                            )
                                        ), 
                                        dbc.Row([
                                            dbc.Col([
                                                dcc.Graph(id = 'graph6', className = 'dbc', config = config_graph), 
                                            ], sm = 12, md = 12)
                                        ], justify = 'center')
                                    ])
                                ], style = tab_card)
                            ]), 
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        dbc.Row(
                                            dbc.Col(
                                                html.H4('Frequência de Palavras nos Feedbacks Negativas', style = {'textAlign':'center'})
                                            )
                                        ),
                                        dbc.Row([
                                            dbc.Col([
                                                dcc.Graph(id = 'graph8', className = 'dbc', config = config_graph), 
                                            ], sm = 12, md = 12)
                                        ], justify = 'center')
                                    ])
                                ], style = tab_card)
                            ]), 
                        ], className='g-2 my-auto', style={'margin-top': '7px'}),
                    ], sm=12, lg=7),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row(
                                    dbc.Col(
                                        html.H4('Distribuição de Palavras nos Feedbacks', 
                                                style = {'textAlign':'center'})
                                    )
                                ),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id = 'graph7', className = 'dbc', config = config_graph, style={'margin-top':'40px'})
                                    ], sm = 12, md = 12)
                                ], justify = 'center')
                            ])
                        ], style = tab_card)
                    ], sm = 12, lg = 5), 
                    
                ], className =' g-3 my-auto', style = {'margin-top':'3px'}), 
            ], fluid=True, style={'height': '-40vh', 'margin-bottom':'30px'})
        ])
    

'''============================# Gráfico 1 #==================================='''
def grafico_1(df):
    # Carregar a imagem como máscara
    mask = np.array(Image.open("assets/nuvem6.png").convert("L"))

    # Transformar em uma máscara binária (0 para áreas vazias, 255 para preenchidas)
    mask = np.where(mask > 128, 255, 0)

    # Filtra os feedbacks positivos e negativos
    df_pos = df [df["Sentiment Label"] == 1 ] ["Feedback_PT"]
    # df_neg = df[df["Sentiment Label"] == 0]["Feedback_PT"]

    # Lista personalizada de stopwords
    stopwords_personalizadas = set(STOPWORDS)
    stopwords_personalizadas.update(["de", "ou", "o", "as", "para", "há", 
                                    "um", "uma", "dos","das", "com", "que", 
                                    "ele", "como", "da", "é", "nos", "aos", 
                                    "mais", "", "seu","sua", "já", "ter", "ser", 
                                    "vai", "está", "são", "", "sim", 'os', 'eu', 
                                    'por','eles', 'e', 'em', 'ao', 'se'
                                    ])
    
    # Paleta de cores personalizada (azul e roxo)
    colors = ['#0d0887', '#5b02a3', '#9a179b', 'orange', 'yellow']  # Tons de azul e roxo

    # Função de cor para as palavras
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return np.random.choice(colors)

    # Função para gerar nuvem de palavras com máscara
    def generate_wordcloud(text):
        wordcloud = WordCloud(
            width=1500,  # Aumentar a largura
            height=800,  # Aumentar a altura
            background_color="#F8BBD0",
            stopwords=stopwords_personalizadas,
            color_func = color_func,
            max_words=150,
            min_font_size=20,
            max_font_size=150,
            contour_color='#0d0887',
            contour_width=10, 
            mask=mask  # Adiciona a máscara da imagem
        ).generate(" ".join(text))

        # Converter a wordcloud em imagem
        wordcloud_image = wordcloud.to_image().convert("RGB")
        
        # Convertendo a imagem para um formato que o Plotly possa exibir
        fig = px.imshow(np.array(wordcloud_image))
        fig.update_layout(
            main_config,
            autosize=True,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),  
            xaxis=dict(
                visible=False, 
                showticklabels=False
                       ),
            yaxis=dict(
                visible=False, 
                showticklabels=False
                ),
            plot_bgcolor="#F8BBD0",  # Cor de fundo do gráfico
            paper_bgcolor="#F8BBD0",  # Cor de fundo da área externa
            hovermode=False
            
        )
    
        return fig

    fig = generate_wordcloud(df_pos)
    return fig

'''============================# Gráfico 2 #==================================='''
def grafico_2(df):
    # Carregar a imagem como máscara
    mask = np.array(Image.open("assets/nuvem6.png").convert("L"))

    # Transformar em uma máscara binária (0 para áreas vazias, 255 para preenchidas)
    mask = np.where(mask > 128, 255, 0)

    # Filtra os feedbacks positivos e negativos
    df_neg = df[df["Sentiment Label"] == 0]["Feedback_PT"]

    # Lista personalizada de stopwords
    stopwords_personalizadas = set(STOPWORDS)
    stopwords_personalizadas.update(["de", "ou", "o", "as", "para", "há", 
                                    "um", "uma", "dos","das", "com", "que", 
                                    "ele", "como", "da", "é", "nos", "aos", 
                                    "mais", "", "seu","sua", "já", "ter", "ser", 
                                    "vai", "está", "são", "", "sim", 'os', 'eu', 
                                    'por','eles', 'e', 'em', 'ao', 'se'
                                    ])
    
    # Paleta de cores personalizada (azul e roxo)
    colors = ['#0d0887', '#5b02a3', '#9a179b', '#e16462', 'yellow']  # Tons de azul e roxo

    # Função de cor para as palavras
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return np.random.choice(colors)

    # Função para gerar nuvem de palavras com máscara
    def generate_wordcloud(text):
        wordcloud = WordCloud(
            width=1500,
            height=800,
            background_color="#F8BBD0",
            stopwords=stopwords_personalizadas,
            color_func = color_func,
            max_words=150,
            min_font_size=20,
            max_font_size=150,
            contour_color='orange',
            contour_width=10,
            mask=mask  # Adiciona a máscara da imagem
        ).generate(" ".join(text))

        # Converter a wordcloud em imagem
        wordcloud_image = wordcloud.to_image().convert("RGB")
        
        # Convertendo a imagem para um formato que o Plotly possa exibir
        fig = px.imshow(np.array(wordcloud_image))
        fig.update_layout(
            main_config,
            autosize=True,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),  # Zerar todas as margens
            xaxis=dict(
                visible=False, 
                showticklabels=False,
                ),
            yaxis=dict(
                visible=False, 
                showticklabels=False, 
                ),
            plot_bgcolor="#F8BBD0",  # Cor de fundo do gráfico
            paper_bgcolor="#F8BBD0",   # Cor de fundo da área externa
            hovermode=False, 
            
        )
    
        return fig

    fig = generate_wordcloud(df_neg)
    return fig

'''============================# Gráfico 3 #==================================='''
def grafico_3 (df):
    ratings = df['Ratings'].value_counts().reset_index()
    ratings.columns = ['ratings', 'count']


    colors = [px.colors.sequential.Plasma[1]]

    fig = px.bar(ratings, 
                x = 'ratings', 
                y = 'count', 
                text = 'count', 
                color_discrete_sequence=colors, 
                )
    
    fig.update_traces(hovertemplate="Avaliação:<b>%{label}</b><br>Valor: %{value}")
    fig.update_layout(
        main_config, 
        height = 230,
        showlegend=False, 
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',        
        yaxis=dict(
            showgrid=True,  
            zeroline=False,  
            showline=False, 
            showticklabels=False,   
            title='Contagem', 
            title_font=dict(color='brown'), 
            linecolor='brown',  
            linewidth=2,
            gridcolor='brown', 
            griddash="dot", 
            gridwidth=1,
        ), 
        xaxis=dict(
            zeroline=False,  
            visible=True,
            showline=True,
            linecolor='brown',  
            linewidth=2,
            title='Avaliações', 
            title_font=dict(color='brown'), 
            tickfont=dict(color='brown'),
            tickcolor='blue', 
            tickwidth=2,
            ticklen=2,
            zerolinecolor='red', 
        ), 
    )


    return fig

'''============================# Gráfico 4 #==================================='''
def grafico_4(df):
    df_c = df.copy()
    # Ajustando o Sentiment Label com base nas Ratings
    # As avaliações 4 e 5 são positivas, a 3 é neutra, e as outras são negativas
    df_c['Sentiment Label'] = df_c['Ratings'].apply(
        lambda x: 'positivos' if x in [4, 5] else ('neutros' if x == 3 else 'negativos')
    )

    sentiment_counts = df_c['Sentiment Label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment Label', 'Count']

    # Gerar uma paleta de cores "plasma"
    colors = [
        px.colors.sequential.Plasma[1], 
        px.colors.sequential.Plasma[4], 
        px.colors.sequential.Plasma[9]
        ]

    # Criar gráfico de pizza com um efeito 3D simulado
    fig = px.pie(sentiment_counts, names='Sentiment Label', 
                 values='Count', 
                 hole = 0.4, 
                 color_discrete_sequence=colors
                 )
    fig.update_traces(pull=[0.03, 0.03, 0.03, 0.03], hovertemplate="Sentimento:<b>%{label}</b><br>Valor: %{value}")  # Ajuste para destacar as fatias
    fig.update_layout(
        main_config,
        height = 230, 
        legend = dict( 
            orientation="h",
            font = dict(color = 'brown')
            
        )

    )

    return fig

'''============================# Gráfico 5 #==================================='''
def grafico_5(df):
    # copiando o dataframe para outra variavel  
    df_c = df.copy()
    
    # Ajustando o Sentiment Label com base nas Ratings
    # As avaliações 4 e 5 são positivas, a 3 é neutra, e as outras são negativas
    df_c['Sentiment Label'] = df_c['Ratings'].apply(
        lambda x: 'positivo' if x in [4, 5] else ('neutro' if x == 3 else 'negativo')
    )

    # Contagem das ocorrências de Ratings para cada Sentiment Label
    df_grouped = df_c.groupby("Sentiment Label")["Ratings"].value_counts().reset_index(name="count")

    # Criando o gráfico de barras
    fig = go.Figure()

    custom_colors = [
        px.colors.sequential.Plasma[4], 
        px.colors.sequential.Plasma[3], 
        px.colors.sequential.Plasma[9], 
        px.colors.sequential.Plasma[0], 
        px.colors.sequential.Plasma[1],
    ]

    # Adicionando os traços para cada Rating
    ratings = df_grouped['Ratings'].unique()
    for i, rating in enumerate(ratings):
        df_rating = df_grouped[df_grouped['Ratings'] == rating]
        fig.add_trace(go.Bar(
            x=df_rating['Sentiment Label'], 
            y=df_rating['count'], 
            name=f"{rating}",  
            marker=dict(color=custom_colors[i % len(custom_colors)]),    
            text=df_rating['count'],  
            textposition='inside',  
            insidetextanchor='middle',  
        ))

    fig.update_traces(hovertemplate="Sentimento:<b>%{label}</b><br>Valor: %{value}")
    fig.update_layout(
        main_config, 
        height =180, 
        barmode="stack",  
        legend=dict(
            traceorder='normal',
            itemsizing='constant',  
            title="Avaliações", 
            title_font = dict(color = 'brown'),
            font = dict(size=13, family='Arial', color='brown'),
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2],  
            ticktext=['negativos', 'neutros', 'positivos'],  
            tickfont=dict(color='brown'), 
            zeroline=False,  
            visible=True,
            showline=True,
            linecolor='brown',  
            linewidth=2,
            title_font=dict(color='brown'), 
            

        ),
        yaxis=dict(
            showline=False,
            showgrid=False,  
            zeroline=False,  
            showticklabels=False, 
            title_font=dict(color='brown'), 
            title = '',
        )
    )

    return fig

'''============================# Gráfico 6 #==================================='''
def grafico_6(df):
    
    def analyze_sentiment_reviews(df):
        # Filtrar avaliações positivas e negativas
        positive_reviews = df[df['Sentiment Label'] == 1].copy()
        negative_reviews = df[df['Sentiment Label'] == 0].copy()

        # Função para pré-processamento dos textos
        def preprocess_text(text):
            # Converter para minúsculas
            text = text.lower()
            # Remover stopwords
            stop_words = set(stopwords.words('portuguese'))
            words = [word for word in text.split() if word not in stop_words]
            return " ".join(words)

        # Aplicar o pré-processamento nas avaliações
        positive_reviews['cleaned_feedback'] = positive_reviews['Feedback_PT'].apply(preprocess_text)
        negative_reviews['cleaned_feedback'] = negative_reviews['Feedback_PT'].apply(preprocess_text)

        # Contar frequência das palavras
        def get_most_common_words(text_series, top_n=10):
            all_words = " ".join(text_series).split()
            word_counts = Counter(all_words)
            return word_counts.most_common(top_n)

        # Obter palavras mais comuns e suas frequências
        positive_common_words = get_most_common_words(positive_reviews['cleaned_feedback'])
        negative_common_words = get_most_common_words(negative_reviews['cleaned_feedback'])

        return positive_common_words, negative_common_words

    # Supondo que você já tenha um DataFrame 'df' definido
    positive_common_words, negative_common_words = analyze_sentiment_reviews(df)

    # Criar DataFrame a partir das palavras comuns positivas
    positive_common_words_df = pd.DataFrame(positive_common_words, columns=['Palavras', 'Frequencia'])

    # Criando o gráfico
    fig = go.Figure()

    # Adicionando barras para palavras positivas
    fig.add_trace(go.Bar(
        x=positive_common_words_df['Frequencia'],
        y=positive_common_words_df['Palavras'],
        name="Positivas",
        marker_color=px.colors.sequential.Plasma[1], 
        text=positive_common_words_df['Frequencia'], 
        orientation='h', 
    ))

    fig.update_traces(hovertemplate="<b>%{label}</b><br>Valor: %{value}") 
    fig.update_layout(
        main_config, 
        height = 250, 
        barmode="group",
        font = dict(size = 15), 
        yaxis = dict(
            tickfont=dict(color='brown'),
            title = ''
        ), 
        xaxis = dict(
            title="Frequência",
            showgrid = False, 
            title_font = dict(color='brown'),
            tickfont=dict(color='brown'),

        ),     
    )

    return fig

'''============================# Gráfico 7 #==================================='''
def grafico_7 (df):

    # Supondo que df já está carregado e tem as colunas necessárias
    df['word_count'] = df['Feedback_PT'].apply(lambda x: len(str(x).split()))

    # Separando feedbacks positivos e negativos
    positive_feedbacks = df[df['Sentiment Label'] == 1]
    negative_feedbacks = df[df['Sentiment Label'] == 0]

    # Substituindo o outlier pela mediana
    median_negative = negative_feedbacks['word_count'].median()
    negative_feedbacks.loc[negative_feedbacks['word_count'] > 100, 'word_count'] = median_negative

    # Criando um intervalo de contagem de palavras
    positive_counts = positive_feedbacks['word_count'].value_counts().sort_index()
    negative_counts = negative_feedbacks['word_count'].value_counts().sort_index()

    # Cálculo da média
    mean_positive = positive_feedbacks['word_count'].mean()
    mean_negative = negative_feedbacks['word_count'].mean()

    # cores das linhas
    positive_colors = px.colors.sequential.Plasma[0]
    negative_colors = px.colors.sequential.Plasma[9]


    # Criando a figura
    fig = go.Figure()

    # Adicionando a linha para feedbacks positivos
    fig.add_trace(go.Scatter(x=positive_counts.index, y=positive_counts.values, mode='lines',
                            line=dict(color=positive_colors,  width=3), name='Positivos', 
                            hovertemplate='Número de palavras: %{x}<br>Positivo: %{y}<extra></extra>', 
                            
                            ))
    # Adicionando a sombra para feedbacks positivos com maior deslocamento e largura
    fig.add_trace(go.Scatter(x=positive_counts.index, y=positive_counts.values + 1.0,  
                            line=dict(color='rgba(44, 3, 145, 0.2)', width=10),  
                            showlegend=False,  hoverinfo='none'))

    # Adicionando a linha para feedbacks negativos
    fig.add_trace(go.Scatter(x=negative_counts.index, y=negative_counts.values, mode='lines',
                            line=dict(color=negative_colors, width=3), name='Negativos',
                            hovertemplate='Negativo: %{y}<extra></extra>'
                            ))
    # Adicionando a sombra para feedbacks negativos com maior deslocamento e largura
    fig.add_trace(go.Scatter(x=negative_counts.index, y=negative_counts.values + 1.0,
                            line=dict(color='rgba(255, 204, 10, 0.3)', width=10),  
                            showlegend=False, 
                            hoverinfo='none'))

    # Ajustes finais no layout
    fig.update_layout(
        main_config, 
        height=400,
        legend_title="Feedback",
        hovermode='x unified',
        yaxis = dict(
            title="Frequência",
            gridcolor='brown', 
            griddash='dot', 
            gridwidth=1,
            zerolinecolor = 'brown', 
            tickfont=dict(color='brown'),
            title_font = dict(color='brown'), 
            showgrid=True, 
            visible=True,
            showline=False, 
            zeroline=False, 
            linecolor='brown',
            linewidth=2,
        ), 
        xaxis =dict(
            title="Número de Palavras",
            showspikes=True, 
            spikemode='across', 
            spikecolor='brown', 
            zeroline=False,  
            visible=True,
            showline=True,
            showgrid = False, 
            linecolor='brown',  
            linewidth=2,
            tickfont=dict(color='brown'),
            title_font = dict(color='brown'), 
        ), 
        legend = dict(
            title = 'Feedbacks', 
            title_font = dict(color = 'brown'),
            font = dict(size=13, family='Arial', color='brown'),

        ), 
        hoverlabel=dict(
            bgcolor='rgba(57, 11, 59, 0.99)',  
            font_size=12,            
            font_color="white",             
        )
    )

    fig.add_annotation(
        text=f"Média de Palavras Positivos: {mean_positive:.0f}",
        xref='paper', yref='paper',
        x=0.98, y=0.80,
        font=dict(size=12, color=px.colors.sequential.Plasma[9]),
        align='center',
        bgcolor='rgba(10, 10, 100, 0.5)',
        showarrow=False
    )

    fig.add_annotation(
        text=f"Média de Palavras Negativos: {mean_negative:.0f}",
        xref='paper', yref='paper',
        x=1.12, y=0.65,
        font=dict(size=12, color=px.colors.sequential.Plasma[0]),
        align='center',
        bgcolor='rgba(255, 204, 0, 0.5)',
        showarrow=False
    )


    return fig

# ============================# Gráfico 7 #===================================
def grafico_8(df):
    
    def analyze_sentiment_negative(df):
        # Filtrar avaliações positivas e negativas
        negative_reviews = df[df['Sentiment Label'] == 0].copy()

        # Função para pré-processamento dos textos
        def preprocess_text(text):
            # Converter para minúsculas
            text = text.lower()
            # Remover stopwords
            stop_words = set(stopwords.words('portuguese'))
            words = [word for word in text.split() if word not in stop_words]
            return " ".join(words)

        # Aplicar o pré-processamento nas avaliações
        negative_reviews['cleaned_feedback'] = negative_reviews['Feedback_PT'].apply(preprocess_text)

        # Contar frequência das palavras
        def get_most_common_words(text_series, top_n=10):
            all_words = " ".join(text_series).split()
            word_counts = Counter(all_words)
            return word_counts.most_common(top_n)

        # Obter palavras mais comuns e suas frequências
        negative_common_words = get_most_common_words(negative_reviews['cleaned_feedback'])

        return negative_common_words

    # Supondo que você já tenha um DataFrame 'df' definido
    negative_common_words = analyze_sentiment_negative(df)

    # Criar DataFrame a partir das palavras comuns positivas
    negative_common_words_df = pd.DataFrame(negative_common_words, columns=['Palavras', 'Frequencia'])

    # Criando o gráfico
    fig = go.Figure()

    # Adicionando barras para palavras positivas
    fig.add_trace(go.Bar(
        x=negative_common_words_df['Frequencia'],
        y=negative_common_words_df['Palavras'],
        name="Negativas",
        marker_color=px.colors.sequential.Plasma[8], 
        text=negative_common_words_df['Frequencia'], 
        orientation='h', 
    ))

    fig.update_traces(marker=dict(line=dict(color=px.colors.sequential.Plasma[0], width=1, )),            hovertemplate="<b>%{label}</b><br>Valor: %{value}") 
    fig.update_layout(
        main_config, 
        height = 250, 
        xaxis_title="Frequência",
        yaxis_title="",
        barmode="group",
        font = dict(size = 15), 
        yaxis = dict(
            tickfont=dict(color='brown'),
        ), 
        xaxis = dict(
            showgrid = False, 
            title_font = dict(color='brown'),
            tickfont=dict(color='brown'),
        ),     
    )

    return fig


'''============================# callbacks #==================================='''
@app.callback(
    Output('graph1', 'figure'),
    Output('graph2', 'figure'), 
    Output('graph3', 'figure') ,
    Output('graph4', 'figure'),
    Output('graph5', 'figure'),
    Output('graph6', 'figure'),
    Output('graph7', 'figure'),
    Output('graph8', 'figure'),
    Input('dummy-input', 'value')  
)
def update_graph(dummy_value):
    return (
        grafico_1(df), 
        grafico_2(df), 
        grafico_3(df), 
        grafico_4(df), 
        grafico_5(df), 
        grafico_6(df),
        grafico_7(df), 
        grafico_8(df)
            )



if __name__ == "__main__":
    # app.run_server(port=8150, debug=True)
    app.run_server(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))