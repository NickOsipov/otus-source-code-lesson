"""
Main Streamlit app.
"""

import streamlit as st


def main():
    """
    Main function of the Streamlit app.
    """
    st.title("Hello World Streamlit")

    name = st.text_input("Input your name:")

    if st.button("Say Hello"):
        if name:
            st.write(f"Hello, {name}!")
        else:
            st.write("Please input your name!")


if __name__ == "__main__":
    main()
