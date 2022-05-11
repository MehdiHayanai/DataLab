from operator import mod
from time import time
from requests import session
import streamlit as st
from ComponentsDataLab import (
    MenuItem,
    info_container,
    StickerComponent,
)
import plotly.graph_objects as go
from io import StringIO
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression




# DATA DOWNLOAD


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

# END DATA DOWNLOAD


##################################################################### STYLING #####################################################################

st.set_page_config(
     page_title="DataLab | GI-IADS tool",
     page_icon="🔬",
     layout="wide",
 )


def missing_values_styling(element):
    return ["background-color: #f0ad4e" if i else "" for i in element]


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("./styles/style.css")

##################################################################### STATES #####################################################################


## Session State
if "actiave_page" not in st.session_state:
    st.session_state["actiave_page"] = "Home"
    st.session_state["data_state"] = 0
    st.session_state["data"] = pd.DataFrame()
    st.session_state["tmp_pca_df"] = None
    st.session_state["clean_df"] = None
    st.session_state["progress"] = None 
    st.session_state["clean"] = False
    st.session_state["models"] = ('SVC', 'RandomForestClassifier', 'Gaussian Naive Bayes', "Knn", "LogisticRegression")




##################################################################### CALLBACKS #####################################################################

## -> Cleaning
def cleaning_callback(action_query, option_query, extra):

        if action_query == "Delete column":
            try:
                st.session_state["clean_df"].drop([option_query], axis=1, inplace=True)
                st.success("✔️ {} was deleted successfully.".format(option_query))
            except:
                st.error("❌ {} was already deleted.".format(option_query))

        elif action_query == "Drop Null rows":
            try:
                st.session_state["clean_df"].dropna(inplace=True)
                st.success("✔️ Null rows were deleted successfully.")
            except:
                st.error("❌ An error occurred try again.")

        elif action_query == "Reset":
            st.session_state["clean_df"] = st.session_state["data"].copy()
            st.success("✔️ Reseted successfully.")

        else :
            try:
                type_transorm = type(st.session_state["clean_df"][option_query][0])
                extra = type_transorm(extra)
                st.session_state["clean_df"][option_query].fillna(extra, inplace=True)
                st.success("✔️ All null values in {} were replaced with {} successfully.".format(option_query, extra))            
            except:
                st.error("❌ Make sure that the data type fits the selected column.")

def cleaning_progress():
    df_size = st.session_state["clean_df"].size
    df_missing_count = st.session_state["clean_df"].isna().sum().sum()
    progess_level = (df_size - df_missing_count) / df_size
    st.session_state["clean"] = progess_level == 1.0
    return progess_level


## -> PCA

def pca_callback(X, n_components, scale):
    """Scales the data using MinMax Scaler and returns a scatter plot."""
    try:
        if scale:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        pca = PCA(n_components)
        pca.fit(X)

    except:
        st.error("❌ A feature might not be supported, please select numerical features only.")
        st.error("❌ Missing values can rise this error, use the cleaning section to handle missing values.")
        return 
    columns_pca = [f"PCA-{i}" for i in range(1,n_components+1)]
    ratio_cumul = pca.explained_variance_ratio_.cumsum()
    pca_df = pd.DataFrame(columns=columns_pca)
    for col, val, cumul_val in zip(columns_pca, pca.explained_variance_ratio_, ratio_cumul):
        pca_df[col] = [val *100, cumul_val *100]

    st.write(pca_df)

    x_trans = pca.transform(X)
    st.session_state["tmp_pca_df"] = st.session_state["clean_df"].copy()
    for ind, column_pca in enumerate(columns_pca):
        st.session_state["tmp_pca_df"][column_pca] = x_trans[: , ind]



    fig = go.Figure(data=[
        go.Bar(name='Variance', x=pca_df.columns, y=pca_df.iloc[0])
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.add_trace(go.Scatter(
        x=pca_df.columns,
        y=pca_df.iloc[1],
        name='Ratio cumulative sum',
    ))
    fig.update_layout(
        autosize=False,
        width=300,
        height=320,

    )

    return fig

def pca_data_maker(X , n_components):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components)
    return pca.fit_transform(X)
    



def plot_pca_callback(X,n_components, target):
    try:
        x_trans = pca_data_maker(X, n_components)

    except:
        st.error("❌ A feature might not be supported, please select only numerical features")
        st.error("❌ Missing values can rise this error use the cleaning section to handle missing values")
        return 
    columns_pca = [f"PCA-{i}" for i in range(1,n_components+1)]
    tmp_pca_df = st.session_state["clean_df"].copy()
    

    for ind, column_pca in enumerate(columns_pca):
        tmp_pca_df[column_pca] = x_trans[: , ind]



    if x_trans.shape[1] == 2:
        fig = px.scatter(tmp_pca_df, x="PCA-1", y="PCA-2", color=target)

    else: 
        fig = px.scatter_3d(tmp_pca_df, x='PCA-1', y='PCA-2', z='PCA-3',
                color=target)
        
    
    return fig

##################################################################### MODELS #####################################################################


def get_model(model_option):
    'SVC', 'RandomForestClassifier', 'Gaussian Naive Bayes', 'Knn', 'LogisticRegression'
    if model_option == "SVC":
        model = SVC()
    
    if model_option == "RandomForestClassifier":
        model = RandomForestClassifier()

    if model_option == "Gaussian Naive Bayes":
        model = GaussianNB()

    if model_option == "Knn":
        model = KNeighborsClassifier()

    if model_option == "LogisticRegression":
        model = LogisticRegression()
        
    
    return model



## DATALAB VIEW


## Navigation section

st.session_state["actiave_page"] = MenuItem()

##################################################################### HOME SECTION #####################################################################


if st.session_state["actiave_page"] == "Home":

    st.markdown(
        """<h2>Welcome to Da<span class="dtlb-different">Lab</span></h2>
            <p>DataLab is a GI-IADS class project made by <b>Mehdi Hayani Mechkouri</b> and <b>Salma Kassimi</b>.<br>GI-IADS : Industrial engineering / data science and artificial intelligence specialty.</p>
        """,
        unsafe_allow_html=True,
    )

    data_loading_section, dataFrame_display_section = st.columns(2)

    with data_loading_section:

        st.markdown(
            f"""<h2>Upload a csv file</h2>
        """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader("")
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()

            # To convert to a string based IO:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

            # To read file as string:
            string_data = stringio.read()

            # Can be used wherever a "file-like" object is accepted:
            try:
                st.session_state["data"] = pd.read_csv(uploaded_file)
                st.session_state["describe"] = st.session_state["data"].describe()
                st.session_state["clean_df"] = st.session_state["data"].copy()
                st.session_state["tmp_pca_df"] = st.session_state["data"].copy()
                st.session_state["data_state"] = 1
            except:
                st.session_state["data_state"] = 0
                st.error("❌ Wrong format, DataLab suports only csv files.")

    with dataFrame_display_section:
        st.markdown(
            f"""<h2>Explore your Data Frame</h2>
        """,
            unsafe_allow_html=True,
        )
        if st.session_state["data_state"] == 0:
            pass
        elif st.session_state["data_state"] == -1:
            pass
        else:
            st.dataframe(st.session_state["data"])

    if st.session_state["data_state"] == 1:
        df_shape, df_columns, explore_cleaning = st.columns(3)
        with df_shape:
            st.markdown("""<h3 class="dtlb-data-section1">Data types</h3>""", unsafe_allow_html=True)
            StickerComponent(st.session_state["data"].dtypes)

        with df_columns:
            st.markdown("""<h3 class="dtlb-data-section">Data columns</h3>""", unsafe_allow_html=True)
            
            StickerComponent(st.session_state["data"].columns)
        with explore_cleaning:
            lignes, columns = st.session_state["data"].shape
            st.markdown("""<h3 class="">Next step</h3>""", unsafe_allow_html=True)
            st.write("DataFrame shape {} X {}".format(lignes, columns))
            st.write("Click on the clean button to continue")
            st.dataframe(st.session_state["describe"], width=400)

    st.markdown("## Learn more about DataLab")
    data_lab_info_section = info_container()

##################################################################### NAVIGATION CLOSE #####################################################################
elif st.session_state["actiave_page"] != "Home" and st.session_state["data_state"] == 0:
    st.error("❌ You need to load your data to access this page")

##################################################################### CLEANING SECTION #####################################################################    
elif st.session_state["actiave_page"] == "Cleaning":
    ### ALGO SECTION ####

    # COPY OF DATA FRAME
    df = st.session_state["data"].copy()
    missing_val_perc = []

    # calculate the percentage of none missing values
    for column in df.columns:
        missing_val_perc.append((1 - (df[column].isnull().sum()) / df.shape[0]) * 100)
    missing_analysis = pd.DataFrame(
        zip(df.columns, missing_val_perc), columns=["col_name", "not_missing"]
    )

    # FIGURE
    fig = go.Figure(
        data=[go.Bar(x=missing_analysis.col_name, y=missing_analysis.not_missing)],
        layout_title_text="The pourcentage of values presence",
    )
    fig.update_layout(barmode='group')
    clean_df_shape = st.session_state["clean_df"].shape
    # END FIGURE 



    expolre_section, query_section, query_output = st.columns(3)

    with expolre_section:

        # EXPLORATION SECTION

        st.markdown("#### EXPLORATION SECTION")

        with st.expander("Explore missing values"):
            # pourcentage of none missing values
            fig.update_layout(template="ggplot2")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Explore nan DataFrame"):
            # colors in yellow none missing values
            st.write(df.isnull().style.apply(missing_values_styling))

        with st.expander("Explore columns Variance"):
            # st.write("### Variance table")
            vals_data = st.session_state["describe"].iloc[2].to_list()
            vals_columns = st.session_state["data"].columns
            vals_dataframe = pd.DataFrame()
            for col, val in zip(vals_columns, vals_data):
                vals_dataframe[col] = [val, None]

            st.dataframe(vals_dataframe, width=1400)

    with query_section:
        st.markdown("#### QUERY SECTION")
        option_query = st.selectbox("Column seletion", df.columns, key="QueryHandler1")

        try:
            ## ADD %2f
            stats_element = [
                "{} {:.2f}".format(
                    mesure, st.session_state["describe"][option_query][mesure]
                )
                for mesure in ["count", "mean", "std", "max", "50%"]
            ]
            missing_text = "missing " + str(st.session_state["clean_df"][option_query].isna().sum())
            stats_element.append(missing_text)
            StickerComponent(stats_element)
        except:
            st.warning("⚠️ Query is only available for numerical features")

    with query_output:
        action_query = st.selectbox(
            "Query action", ["Delete column", "Fill Null values", "Drop Null rows", "Reset"],
        )
        fill_na_value = None
        if action_query == "Fill Null values":
            fill_na_value = st.text_input('New value', '0')

        confirm_action = st.button("Confirme action")

        if confirm_action:
            cleaning_callback(action_query, option_query, fill_na_value)
        cleaning_pourcentage = int(cleaning_progress()*100)
        st.write(f"Cleaning {cleaning_pourcentage}%")
        st.session_state["progress"] = st.progress(cleaning_pourcentage)
            


    try:
        graphe_section = st.container()
        with graphe_section:

            st.write("### DISTRIBUTION SECTION")

            template_plot = "plotly"
            try:
                fig_option_distribution = go.Figure()
                
                fig_option_distribution.add_trace(go.Histogram(x=st.session_state["clean_df"][option_query], name='Cleaned DataFrame'))
                fig_option_distribution.add_trace(go.Histogram(x=st.session_state["data"][option_query], name='Original DataFrame'))
                fig_option_distribution.update_layout(
                    title=f"{option_query} Histogram",
                    xaxis_title=f"{option_query}",
                    yaxis_title="Count",
                    template= template_plot,
                )
                fig_option_distribution.update_traces(opacity=0.75)
                st.plotly_chart(fig_option_distribution, use_container_width=True)


            except:
                st.warning(f"This column was probably deleted")

        col_original_df, col_cleaned_df = st.columns(2)
        with col_original_df:
            st.markdown("## Inital DataFrame")   
            st.write(st.session_state["data"])
        with col_cleaned_df:
            st.markdown("## New DataFrame")   
            st.write(st.session_state["clean_df"])
    except:
        st.warning("⚠️ An error occurred, try changing your query option")
##################################################################### PCA SECTION #####################################################################
elif st.session_state["actiave_page"] == "PCA":
    pca_option, pca_features = st.columns(2)

    st.session_state["set"]  = set(st.session_state["clean_df"].columns)

    with pca_option:
        st.markdown("""<h3 class="dtlb-data-section1">PCA Commandes</h3>""", unsafe_allow_html=True)
        st.session_state["X"] = st.multiselect(
        'Select your features',
        st.session_state["set"],
        st.session_state["set"])

        st.session_state["Y"] = st.selectbox(
        'Target', st.session_state["set"] - set(st.session_state["X"]))
        launch_pca = False
        if len(st.session_state["X"]) > 0:
            n_components = st.slider('Number of principal components', 0, len(st.session_state["X"]),len(st.session_state["X"]) )
            scale = st.checkbox('Scale')

            icon = "🟢" if st.session_state["clean"] else  "🔴" 
            launch_pca = st.button(f"Start {icon}")

    with pca_features:
        if launch_pca:
            X = st.session_state["clean_df"][st.session_state["X"]]
            pca_fig = pca_callback(X, n_components, scale)
            if pca_fig != None:
                st.plotly_chart(pca_fig, use_container_width=True)
            else: 
                pass
        else:
            st.markdown("Waiting for you input ...")
    
    pca_view = st.expander("2D and 3D view")

    with pca_view:
        plot_dim_pca = st.radio("Select a ploting dimension", ("2D", "3D"))
        st.markdown(plot_dim_pca)
        pca_start = st.button("Launch view")
        if pca_start:
            try:
                if plot_dim_pca == "2D" and st.session_state["clean_df"][st.session_state["X"]].shape[1] > 1 and st.session_state["Y"] != None:
                    pca_2d = plot_pca_callback(st.session_state["clean_df"][st.session_state["X"]],int(plot_dim_pca[0]), st.session_state["Y"])
                    st.plotly_chart(pca_2d, use_container_width=True)
                elif  plot_dim_pca == "3D" and st.session_state["clean_df"][st.session_state["X"]].shape[1] > 2 and st.session_state["Y"] != None:
                    pca_3d = plot_pca_callback(st.session_state["clean_df"][st.session_state["X"]],int(plot_dim_pca[0]), st.session_state["Y"])
                    st.plotly_chart(pca_3d, use_container_width=True)
                elif st.session_state["Y"] == None:
                    st.warning("⚠️ Select a target")
                else:
                    st.warning("⚠️ Dimension is too low")
            except:
                st.info("Pick your features and target")
##################################################################### HOME SECTION #####################################################################
elif st.session_state["actiave_page"] == "Models":
    st.session_state["model_set"]  = set(st.session_state["tmp_pca_df"].columns)

    if not st.session_state["clean"]:
        st.warning("⚠️ Your data is not clean")
    model_section, model_param_section = st.columns(2)
    
    with model_section:
        st.write("### MODEL SECTION")
        training_features = st.multiselect(
            'Features',
            st.session_state["model_set"],
            st.session_state["model_set"])
        training_target = st.selectbox(
            'Target', st.session_state["model_set"] - set(training_features))
        training_proportion = st.slider('Training size', 0, 100, 70)

    with model_param_section:
        st.write("### MODEL PARAMETERS")
        model_option = st.selectbox(
        'Select you model',
        st.session_state["models"])

        model = get_model(model_option)

        st.write(model.get_params())

##################################################################### DOWNLOAD SECTION #####################################################################
elif st.session_state["actiave_page"] == "Download":
    try :
        st.title("Final DataFrame")
        tmp_pca_shape = st.session_state["tmp_pca_df"].shape
        st.write(st.session_state["tmp_pca_df"])
        st.caption(f"shape = {tmp_pca_shape}")
        csv = convert_df(st.session_state["tmp_pca_df"])

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f'DataLab{int(time())}.csv',
        )
    except:
        st.warning("⚠️ You haven't used the PCA section")
else :
    st.error("❌ Where are you going ??")