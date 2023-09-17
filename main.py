import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import csv

st.set_page_config(
    page_title="Forecast", initial_sidebar_state="collapsed", page_icon="🔮"
)

tabs = ["Application", "About"]
page = st.sidebar.radio("Tabs", tabs)

st.title("Forecast Application 🧙‍♂️")
st.markdown("### Upload any time-series data to forecast.")


def read_csv_with_auto_delimiter(file):
    try:
        file_content = file.read().decode("utf-8")

        # Use the csv.Sniffer to automatically detect the delimiter
        dialect = csv.Sniffer().sniff(file_content)

        # Create a StringIO object to simulate a file-like object
        from io import StringIO

        file_like = StringIO(file_content)

        # Use the detected delimiter to read the CSV file into a DataFrame
        df = pd.read_csv(file_like, delimiter=dialect.delimiter)

        return df

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


if page == "Application":
    st.markdown("## 1.Upload data 📅")
    with st.expander("Date Format"):
        st.write("The data must contain date-time and value column")
    df_file = st.file_uploader("Upload a csv file", type=["csv"])
    df = pd.DataFrame
    date = "datee"
    value = "valuee"
    flag = 0
    # if st.button("Process"):
    #     # with st.spinner("Loading..."):
    #     if df_file == None:
    #         st.warning("Please upload a csv file")
    #     else:
    if df_file:
        df = read_csv_with_auto_delimiter(df_file)
        if df.shape[0] == 0:
            st.warning("Something went wrong")
        else:
            st.success("File loaded successfully")
            flag = 1
            st.dataframe(df.head())
            st.write(f"Number of rows: {df.shape[0]}")
            st.write("Columns name:")
            st.write(list(df.columns.values))
            # st.line_chart(df.set_index('date'))
            # st.area_chart(df.set_index('date'))

            options = list(df.columns.values)

            col1, col2 = st.columns(2)

            with col1:
                date = st.selectbox("Select date column", options, key="dropdown1")
            with col2:
                value = st.selectbox("Select value column", options, key="dropdown2")

            df[date] = pd.to_datetime(df[date])
            ndf = df.rename(columns={date:"ds",value:"y"})

            visual_ck = st.checkbox("Visualize")
            if visual_ck:
                with st.spinner("Loading"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Dataframe Sample")
                        st.dataframe(df.sample(20))
                    with col2:
                        st.write("Describe")
                        st.dataframe(df.describe())

                    with st.expander("Select colour"):
                        colour_hex = st.text_input(
                            "Enter colour Hex code", placeholder="Ex:#800080"
                        )
                    st.markdown(
                        """
                        <div style="text-align:center;">
                            <h3>Interactive Time series plot</h3>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if len(colour_hex) > 0:
                        try:
                            st.line_chart(
                                data=df,
                                x=date,
                                y=value,
                                use_container_width=True,
                                height=300,
                                color=colour_hex,
                            )
                            # st.line_chart(x=df[date],y=df[value],use_container_width =True,height = 300)
                        except:
                            st.warning("Unable to detect colour")
                            st.t
                            st.line_chart(
                                data=df,
                                x=date,
                                y=value,
                                use_container_width=True,
                                height=300,
                                color="#3cdfff",
                            )
                    else:
                        st.line_chart(
                            data=df,
                            x=date,
                            y=value,
                            use_container_width=True,
                            height=300,
                            color="#3cdfff",
                        )

    st.markdown("## 2.Parameters Configration ⚙️")
    with st.expander("Horizon"):
        st.number_input("")
    st.markdown("## 3.Forecast 🪄")
    if flag:
        if st.checkbox("Train model"):
            with st.spinner("Training"):
                # print(ndf.columns)
                m = Prophet()
                model = m.fit(ndf)
                st.success("Model trained sucessfully")
        if st.checkbox("Generate Forecast(Predict)"):
            with st.spinner("Predicting"):
                future_df = m.make_future_dataframe(periods=365)
                st.success("Prediction generated successfully")
                forecast = m.predict(future_df)
                st.dataframe(forecast)
                fig1 = m.plot(forecast)
                st.write(fig1)
            
        if st.checkbox("Show Components"):
            with st.spinner("Plotting"):
                fig2 = m.plot_components(forecast)
                st.write(fig2)

            
    st.markdown("## 4.Model validation 🦾")
    st.markdown("## 5.Hyperparameter Tuning 🤖")
    st.markdown("## 6.Results ✨")


else:
    st.title("About Page")
