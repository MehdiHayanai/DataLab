from turtle import onclick
from matplotlib.pyplot import axis
import streamlit as st
import streamlit.components.v1 as components
from Components import (
    MenuItem,
    info_container,
    StickerComponent,
)
import plotly.graph_objects as go
import plotly.figure_factory as ff
from io import StringIO
import pandas as pd


# Data handling


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


def missing_values_styling(element):
    return ["background-color: #f0ad4e" if i else "" for i in element]


## Styling

st.set_page_config(layout="wide")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("./styles/style.css")


## Session State
if "actiave_page" not in st.session_state:
    st.session_state["actiave_page"] = "Home"
    st.session_state["data_state"] = 0
    st.session_state["data"] = pd.DataFrame()




## Buttons callbacks 

def cleaning_callback(action_query, option_query, extra):

        if action_query == "Delete column":
            try:
                st.session_state["clean_df"].drop([option_query], axis=1, inplace=True)
                st.success("{} was deleted successfully.".format(option_query))
            except:
                st.error("{} was already deleted.".format(option_query))

        elif action_query == "Drop Null rows":
            try:
                st.session_state["clean_df"].dropna(inplace=True)
                st.success("Null rows were deleted successfully.")
            except:
                st.error("An error occurred try again.")

        elif action_query == "Reset":
            st.session_state["clean_df"] = st.session_state["data"].copy()
            st.success("Reseted successfully.")


        else :
            type_transorm = type(st.session_state["clean_df"][option_query][0])
            extra = type_transorm(extra)
            try:
                st.session_state["clean_df"][option_query].fillna(extra, inplace=True)
                st.success("All null values in {} were replaced with {} successfully.".format(option_query, extra))            
            except:
                st.error("Make sure that the data type fits the selected column.")


## Page handling


## Navigation section

st.session_state["actiave_page"] = MenuItem()

## Infor about DataLab
if st.session_state["actiave_page"] == "Home":

    st.markdown(
        """<h2>Welcome to Da<span class="dtlb-different">Lab</span></h2>
            <p>Build your first machine learning classifier, and set your workflow with DataLab.</p>
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
                st.session_state["data_state"] = 1
            except:
                st.session_state["data_state"] = 0
                st.error("Wrong format, DataLab suports only csv files.")

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
            # print(st.session_state["data"].dtypes)
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

    st.title("You want to learn more about DataLab")
    data_lab_info_section = info_container()
elif st.session_state["actiave_page"] != "Home" and st.session_state["data_state"] == 0:
    st.error("You need to load your data to access this page")
elif st.session_state["actiave_page"] == "Cleaning":
    # Cleaning
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
            StickerComponent(stats_element)
        except:
            st.warning("Query is only available for numerical features")

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

    with st.expander("Cleaned DataFrame"):
        st.write(st.session_state["clean_df"])

    graphe_section = st.container()
    with graphe_section:
 
        st.write("### DISTRIBUTION SECTION")

        try:
            fig_option_distribution = go.Figure(data=[go.Histogram(x=df[option_query])])
            fig_option_distribution.update_layout(
                title=f"{option_query} Histogram",
                xaxis_title=f"{option_query}",
                yaxis_title="Count",
                legend_title="Legend Title",
                template="seaborn",
                # color = 'indianred'
            )
            # fig_option_distribution.update_traces(opacity=0.75, color_discrete_sequence=['indianred'])
            fig_option_distribution.update_traces(opacity=0.75)
            st.plotly_chart(fig_option_distribution, use_container_width=True)

        except:
            st.warning(f"{option_query} is not numerical")

