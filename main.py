import streamlit as st
import pandas as pd
st.set_page_config(
    page_title="Choose your Hotel!",
    page_icon="🧊",
)

# Create a sidebar
st.sidebar.title("Go ahead!")

# Create a dropdown menu for navigation
page_options = ["A little description...","Choose the Hotel", "Statistics", "Comments"]
selected_page = st.sidebar.selectbox("Select a page", page_options)

# Create the pages
if selected_page == "A little description...":
    st.title("A little Description...")
    st.write("Welcome to our application! We created this app based on a webscrapping we did on the Hotel.com Website! Hotel.com is a website that helps you search for the best hotels in the area that you want to stay in. After creating our dataframes, we did a clustering of prices and a Sentimental Analysis on the comments. The destination we chose is Paris!")
    st.image('eff.jpg', width=500)
elif selected_page == "Choose the Hotel":
    st.image('yo.jpg', width=300)
    st.title("Choose your Hotel!")

    st.write("With this app, I'll help you find the best hotel for your stay!")
    df = pd.read_csv("df_final_hotel.csv")
    df_pandas = pd.read_csv("df_pandas.csv")
    df_analyse_sent = pd.read_csv("df_analyse_sent.csv")
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(df)
        st.write(df_pandas)
        st.write(df_analyse_sent)
    selected_hotel = st.selectbox('Select a hotel', df['titre'])

    if selected_hotel:
        selected_row = df.loc[df['titre'] == selected_hotel]
        st.write('Price of the selected hotel: ', selected_row['prix'].values[0],'euros')
        st.write('Rating of the selected hotel: ', selected_row['note'].values[0],'/10')
        st.write('Equipement of the selected hotel: ', selected_row['equipement_2'].values[0])
elif selected_page == "Statistics":
    import plotly.express as px
    st.subheader('Distribution')
    df_pandas = pd.read_csv("df_pandas.csv")
    df = pd.read_csv("df_final_hotel.csv")
    graph = st.radio(
        'Distributions of each caracteristic',
        ("Prices", "Ratings"))
    st.write(f"Here you can see the distribution of the {graph}!")
    if graph== "Prices":
        st.plotly_chart(px.histogram(df, x='prix', range_x=[0,df['prix'].max()],width=700, height=400))
    elif graph == "Ratings":
        st.plotly_chart(px.histogram(df, x='note', range_x=[0,df['note'].max()],width=700, height=400))

    st.subheader("Clustering of prices")
    #Import and fitting of the kmeans model
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    quanti=df_pandas.drop(['titre','link'],axis=1)
    X = df_pandas.drop(['titre','link','Autres_1','Parking','Animaux','Autres_2','Non-fumeurs','Wi-Fi','Unnamed: 0'],axis=1).values
    sc = StandardScaler()
    X_normalise = sc.fit_transform(X)

    km = KMeans(n_clusters=4,random_state=0).fit(X_normalise)
    preds = km.predict(X_normalise)
    df_pandas['cluster'] = preds

    X = sc.inverse_transform(X_normalise)

    clus = st.radio(
        'Clustering',
        ("Cluster", "Centroids"))
    st.write(f"Here you can see the graph of the {clus}!")
    if clus== "Cluster":
        import plotly.graph_objs as go
        trace1 = go.Scatter(x=X[preds == 0, 0], y=X[preds == 0, 1], 
                    mode='markers', name='Cluster 1', 
                    marker=dict(size=20, color='cornflowerblue'))
        trace2 = go.Scatter(x=X[preds == 1, 0], y=X[preds == 1, 1], 
                   mode='markers', name='Cluster 2', 
                   marker=dict(size=20, color='cornsilk'))
        trace3 = go.Scatter(x=X[preds == 2, 0], y=X[preds == 2, 1], 
                   mode='markers', name='Cluster 3', 
                   marker=dict(size=20, color='coral'))
        trace4 = go.Scatter(x=X[preds == 3, 0], y=X[preds == 3, 1], 
                   mode='markers', name='Cluster 4', 
                   marker=dict(size=20, color='crimson'))

        data = [trace1, trace2, trace3,trace4]
        layout = go.Layout(title='Clusters', showlegend=True)
        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig)
    elif clus == "Centroids":
        import plotly.graph_objs as go
        trace = go.Scatter(x=km.cluster_centers_[:, 0], y=km.cluster_centers_[:, 1],
                   mode='markers', name='Centroids',
                   marker=dict(size=20, color='cadetblue'))

        data = [trace]
        layout = go.Layout(title='Centroids', showlegend=True)
        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig)

elif selected_page == "Comments":
    st.subheader("Sentimental Analysis")
#     df_analyse_sent = pd.read_csv("df_analyse_sent.csv")
#     df_analyse_sent['description'] = df_analyse_sent['description'].astype(str)
#     text = ""
#     for comment in df_analyse_sent.description : 
#         text += comment

#     # Importer stopwords de la classe nltk.corpus
#     import nltk
#     from nltk.corpus import stopwords
    
#     # Initialiser la variable des mots vides
#     stop_words = set(stopwords.words('french'))

#     #Importer les packages nécessaires
#     from wordcloud import WordCloud


#     #Importer les packages nécessaires
#     from PIL import Image
#     import numpy as np

#     def plot_word_cloud(text, masque, background_color = "black") :
#     # Définir un masque
#         mask_coloring = np.array(Image.open(str(masque)))

#         # Définir le calque du nuage des mots
#         wc = WordCloud(background_color=background_color, max_words=1000, stopwords=stop_words, mask = mask_coloring,colormap='flag',max_font_size=50, random_state=42)
#         import matplotlib.pyplot as plt
#         # Générer et afficher le nuage de mots
#         plt.figure(figsize= (5,5))
#         wc.generate(text)
#         plt.imshow(wc)
#         plt.axis('off')
#         st.set_option('deprecation.showPyplotGlobalUse', False)
#         st.pyplot()
#         st.set_option('deprecation.showPyplotGlobalUse', False)

#     df_analyse_sent["sentiment"] = 0
#     df_analyse_sent.loc[df_analyse_sent["note"].isin([10,8,6]),'sentiment'] = 1
#     # print(df_analyse_sent.head())
#     df_analyse_sent=df_analyse_sent.drop('note',axis=1)

#     # # Définir les données positives et négatives de types string
#     df_pos = df_analyse_sent[df_analyse_sent.sentiment == 1]
#     df_neg = df_analyse_sent[df_analyse_sent.sentiment == 0]

#     # print(df_pos)
            
#     text_pos = ""
#     for e in df_pos.description : text_pos += e
#     text_neg = ""
#     for e in df_neg.description : text_neg += e

    # Tracer le nuage de mots
    sent = st.radio(
        'Sorting comments between negative and positive',
        ("Positive comments", "Negative comments"))
    st.write(f"Here you can see the most recurent words in the {sent}!")
    if sent== "Positive comments":
        st.image('arc_pos'.jpg,width=500))
#         plot_word_cloud(text_pos, "mrc.jpg", "white")
        st.set_option('deprecation.showPyplotGlobalUse', False)
    elif sent == "Negative comments":
        st.image('rat_neg.jpg',width=500))
#         plot_word_cloud(text_neg, "yat.jpg","white")
        st.set_option('deprecation.showPyplotGlobalUse', False)



import streamlit as st
import numpy as np

