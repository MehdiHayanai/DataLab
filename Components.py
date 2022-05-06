from streamlit_option_menu import option_menu
import streamlit as st


def MenuItem():
    menu_item = option_menu(
        menu_title="DataLab",
        menu_icon="🔬",
        options=[
            "Home",
            "Cleaning",
            "PCA",
            "Download",
        ],
        icons=["house-fill", "file-earmark-excel-fill", "dash-circle-fill", "file-arrow-down-fill",],
        styles={
            "container": {
                "border-radius": "0px",
                "padding": "10px",
                "box-shadow": "0 0 11px rgba(33,33,33,.2)",
            },
            "menu-title": {
                "display": "flex",
                "justify-content": "center",
                "align-items": "center",
                # "background-color": "red",
                "width": "100%",
                "font-size": "40px",
                "font-family": "'Inter', sans-serif",
            },
            "menu-icon": {
                "height": "50px",
                "width": "50px",
                "display": "flex",
                "justify-content": "center",
                "align-items": "center",
            },
            "nav-link-selected": {
                "box-shadow": "0 0 11px rgba(33,33,33,.2)",
                "transition": "box-shadow 2s ease-in-out",
            },
        },
        orientation="horizontal",
    )
    return menu_item


def make_card_html(main_text, description, lib_text, src=None, alt="logo"):
    html_text = f"""
            <div class="card">
            <div class="media"> <img
                    src="{src}" alt="{alt}" width="1000" height="1000" alt="{alt}"> 
            </div>
            <div class="primary-title">
                <div class="primary-text">{main_text}</div>
                <div class="secondary-text">powered <b>by {lib_text}</b></div>
            </div>
            <div class="supporting-text"> {description}
            </div>
        </div>"""
    return html_text


def CardItem(items):
    st_section = st.container()
    html_start = """<div class="cards-wrapper">"""
    for item in items:
        html_start += item
    html_start += "</div>"
    st_section.write(html_start, unsafe_allow_html=True)

    return st_section


def info_container():
    datacleaning_text = make_card_html(
        "DataLab's cleaning solution",
        "DataLab accelerates the data cleaning process and saves the user's preferences and habits related to data handling, thus making the automatization of this step possible in the future.",
        "Pandas",
        "data:image/png;base64,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",
        alt="Pandas logo",
    )

    machine_text = make_card_html(
        "Data Preprocessing",
        "DataLab accelerates the data cleaning process and saves the user preference and habits related to data handling thus making the automatization of this step possible in the future.",
        "Scikit learn",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1024px-Scikit_learn_logo_small.svg.png",
        alt="Scikit learn logo",
    )
    reduction_text = make_card_html(
        "Principal component analysis",
        "DataLab accelerates the data cleaning process and saves the user preference and habits related to data handling thus making the automatization of this step possible in the future.",
        "Scikit learn",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1024px-Scikit_learn_logo_small.svg.png",
        alt="Scikit learn logo",
    )
    # models_text = make_card_html(
    #     "Models creation",
    #     "DataLab accelerates the data cleaning process and saves the user preference and habits related to data handling thus making the automatization of this step possible in the future.",
    #     "Scikit learn",
    #     "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1024px-Scikit_learn_logo_small.svg.png",
    #     alt="Scikit learn logo",
    # )

    new_container = CardItem(
        [datacleaning_text, machine_text, reduction_text]
    )
    return new_container


def NavigationComponent(active_page):
    pages = [
        "Home",
        "Cleaning",
        "Preprocessing",
        "Dim-Reduction",
        "Models",
        "Download",
    ]
    index_of_active_page = -1
    for i, page in enumerate(pages):
        if page == active_page:
            index_of_active_page = i
            break

    if index_of_active_page == 0:
        next_page = st.button(f"Move to {pages[index_of_active_page+1]}>>")
        previous_page = False
    elif index_of_active_page == 5:
        previous_page = st.button(f"Go back to {pages[index_of_active_page-1]}<<")
        next_page = False
    else:
        previous_page = st.button(f"Go back to {pages[index_of_active_page-1]}<<")
        next_page = st.button(f"Move to {pages[index_of_active_page+1]}>>")

    if previous_page:
        return pages[index_of_active_page - 1]
    elif next_page:
        return pages[index_of_active_page + 1]
    else:
        return pages[index_of_active_page]


def StickerComponent(column_elements):

    extra_add = {"int64": "dtlb-int", "float64": "dtlb-float", "object": "dtlb-object"}
    add_class = ""
    html_text = '<div class="dtlb-sticker-container">'

    for col in column_elements:
        if str(col) in extra_add:
            add_class = extra_add[str(col)]
        else:
            add_class = ""
        html_text += f"""<div class="dtlb-sticker">
            <p class="dtlb-elevation {add_class}">
                {col}
            </p>
        </div>
        """

    html_text += "</div>"

    st.write(html_text, unsafe_allow_html=True)


def DataShapeComponent(shape):
    html_text = f"""<div class="dtlb-shape-container">
            <p class="dtlb-elevation dtlb-shape-info">
                {shape[0]}
            </p>
            <p class="dtlb-elevation dtlb-shape-info">
                {shape[1]}
            </p>
        </div>"""

    st.write(html_text, unsafe_allow_html=True)

