import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import holidays
import datetime
import itertools
import json
from prophet.serialize import model_to_json
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
import csv
from country_name import country_set, date, value

# country_set = set()
# date="datee"
# value = "valuee"
# @st.cache_data
# def initialize():
#     country_set = set()
#     date = "datee"
#     value = "valuee"
#     return country_set, date, value

# country_set,date,value = initialize()


# @st.cache_data
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


def get_country_holidays(country_code, year):
    country_holidays = holidays.CountryHoliday(country_code, years=year)
    # print(sorted(country_holidays))
    return sorted(country_holidays)


country_codes = {
    "India": "IN",
    "Italy": "IT",
    "United States": "US",
    "Germany": "DE",
    "Spain": "ES",
    "France": "FR",
}


st.set_page_config(
    page_title="Forecast", initial_sidebar_state="collapsed", page_icon="🔮"
)

tabs = ["Application", "About"]
page = st.sidebar.radio("Tabs", tabs)

st.title("Forecast Application 🧙‍♂️")
st.markdown("### Upload any time-series data to forecast.")
st.header("", divider="rainbow")


if page == "Application":
    st.markdown("## 1.Upload data 📅")
    with st.expander("Date Format"):
        st.write("The data must contain date-time and value column")
    df_file = st.file_uploader("Upload a csv file", type=["csv"])
    df = pd.DataFrame
    # date = "datee"
    # value = "valuee"
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

            options = list(df.columns.values)

            col1, col2 = st.columns(2)

            with col1:
                date = st.selectbox("Select date column", options, key="dropdown1")
            with col2:
                value = st.selectbox("Select value column", options, key="dropdown2")

            df[date] = pd.to_datetime(df[date])
            ndf = df.rename(columns={date: "ds", value: "y"})

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
        prediction_period_param = st.number_input(
            "Enter the number of future periods(days) you want to predict", value=30
        )
    with st.expander("Growth"):
        growth_values = ["linear", "logistic", "flat"]
        growth_param = st.selectbox("Select the growth function", growth_values)
    with st.expander("Trend Component"):
        st.write("Add or remove components")

        daily_param = True if st.checkbox("Daily") else False
        weekly_param = True if st.checkbox("Weekly") else False
        monthly_param = True if st.checkbox("Monthly") else False
        yearly_param = True if st.checkbox("Yearly") else False

    with st.expander("Seasonality mode"):
        seasonality_mode_param = st.radio("Seasonality", ["additive", "multiplicative"])

    # with st.expander("Holidays"):
    #     country_name = st.selectbox("Select a country",["Choose a country","India","United States","Germany","Italy"])
    #     if country_name != "Choose a country":
    #         country_add = st.checkbox("Add country holiday to model",value=(True if country_name in country_set else False))
    #         if country_add:
    #             country_set.add(country_name)
    #         elif country_name in country_set:
    #             country_set.remove(country_name)
    #     st.write(country_set)

    #     holidays_set = set()
    #     for name in country_set:
    #         country_code = country_codes[name]
    #         holiday_list =  get_country_holidays(country_code, datetime.datetime.now().year)
    #         for x in holiday_list:
    #             holidays_set.add(x)
    #             # st.write(x)
    #     holiday_list_df = list(holidays_set)
    #     holiday_df = pd.DataFrame(holiday_list_df,columns=['Date'])
    #     st.dataframe(holiday_df)

    with st.expander("Hyperparameter"):
        st.subheader("Changepoint prior scale")
        st.write(
            """Parameter modulating the flexibility of the
    automatic changepoint selection. Large values will allow many changepoints, small values will allow few changepoints."""
        )
        Changepoint_param = st.number_input("Changepoint prior scale", value=0.05)
        st.subheader("Seasonality prior scale")
        st.write(
            """ Parameter modulating the strength of the
    seasonality model. Larger values allow the model to fit larger seasonal fluctuations, smaller values dampen the seasonality. Can be specified for individual seasonalities using add_seasonality."""
        )
        Seasonality_prior_scale_param = st.number_input(
            "Seasonality prior scale", value=10
        )

    st.markdown("## 3.Forecast 🪄")
    if not flag:
        st.write("Upload data first")
    if st.checkbox("Train model"):
        # try:
        with st.spinner("Training"):
            # print(ndf.columns)
            m = Prophet(
                growth=growth_param,
                daily_seasonality=daily_param,
                weekly_seasonality=weekly_param,
                yearly_seasonality=yearly_param,
                seasonality_mode=seasonality_mode_param,
                changepoint_prior_scale=Changepoint_param,
                seasonality_prior_scale=Seasonality_prior_scale_param,
            )
            # if holidays:
            #     m.add_country_holidays(country_name=[x for x in country_set])

            if monthly_param:
                m.add_seasonality(name="monthly", period=30.4375, fourier_order=5)
            model = m.fit(ndf)
            st.success(
                f"Model trained sucessfully on data upto date { ndf.at[ndf.shape[0]-1,'ds'].date() }"
            )
        # except:
        #     st.warning("Try selecting date and value columns")
    if st.checkbox("Generate Forecast(Predict)"):
        with st.spinner("Predicting"):
            future_df = m.make_future_dataframe(periods=365)
            forecast = m.predict(future_df)
            st.success(
                f"Prediction generated successfully upto date {forecast.at[forecast.shape[0]-1,'ds'].date()}"
            )
            st.dataframe(forecast)
            fig1 = m.plot(forecast)
            st.write(fig1)

    if st.checkbox("Show Components"):
        with st.spinner("Plotting"):
            fig2 = m.plot_components(forecast)
            st.write(fig2)

    st.markdown("## 4.Model validation 🦾")
    with st.expander("Explanation"):
        st.write("X")
    with st.expander("Cross Validation"):
        initial = st.number_input("initial", value=365)
        initial = str(initial) + " days"
        period = st.number_input("period", value=90)
        period = str(period) + " days"
        horizion = st.number_input("horizon", value=90)
        horizon = str(horizion) + " days"
    with st.expander("Metrics"):
        metric = 0
        if st.checkbox("Calculate metrics"):
            df_cv = cross_validation(
                m,
                initial=initial,
                period=period,
                horizon=horizon,
                parallel="processes",
            )
            st.dataframe(df_cv)

            df_p = performance_metrics(df_cv)

            metric = 1
            # df_p = np.array(df_p['horizon'].dt.to_pydatetime())
            # print(df_p.info())
            # print(type(df_p))
            # st.write(df_p)
            # st.dataframe(df_p)
            chosen_metric = st.selectbox(
                "Select metric to plot",
                ["Choose a metric", "mse", "rmse", "mae", "mape", "coverage"],
            )
            if chosen_metric != "Choose a metric":
                if metric:
                    with st.spinner("Plotting..."):
                        fig = plot_cross_validation_metric(df_cv, metric=chosen_metric)
                        st.write(fig)
                else:
                    st.warning("Please calculate metrics first")

    st.markdown("## 5.Hyperparameter Tuning 🤖")
    st.write("Here you can find the best possible combination of hyperparameters")

    param_grid = {
        "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
        "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
    }

    if flag:
        if st.button("Optimize hyperparameters"):
            with st.spinner("Optimizing"):
                # Generate all combinations of parameters
                all_params = [
                    dict(zip(param_grid.keys(), v))
                    for v in itertools.product(*param_grid.values())
                ]
                rmses = []  # Store the RMSEs for each params here

                for params in all_params:
                    m = Prophet(**params).fit(ndf)  # Fit model with given params
                    df_cv = cross_validation(
                        m,
                        initial=initial,
                        period=period,
                        horizon=horizon,
                        parallel="processes",
                    )
                    df_p = performance_metrics(df_cv, rolling_window=1)
                    rmses.append(df_p["rmse"].values[0])

                tuning_results = pd.DataFrame(all_params)
                tuning_results["rmse"] = rmses
                st.write(tuning_results)

                best_params = all_params[np.argmin(rmses)]

                st.write("Best combination of paramater for this data is:")
                st.write(best_params)

                st.write(
                    "You can repeat this process with different parameter configration in configration section"
                )

    st.markdown("## 6.Results ✨")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Export forecast (.csv)'):
            with st.spinner("Exporting.."):
                export_forecast = pd.DataFrame(forecast[['ds','yhat_lower','yhat','yhat_upper']])
                st.dataframe(export_forecast)
                export_forecast= export_forecast.to_csv()
                st.download_button("Download forecast csv",data=export_forecast,file_name="forecast.csv")
                
    with col2:
        if st.button("Export model metrics"):
            with st.spinner("Exporting"):
                try:
                    st.dataframe(df_p)
                    export_metrics = df_p.to_csv()
                    st.download_button("Download metrics csv",data=export_metrics,file_name="metrics.csv")
                except:
                    st.warning("Please calculate metrics first")
    with col3:
        if st.button("Export model"):
            with st.spinner("Exporting"):
                try:
                    model_json = json.dumps(model_to_json(m), indent=4)
                    st.download_button(
                        label="Download Model JSON",
                        data=model_json,
                        file_name="model.json",
                        key="download-model-json"
                    )
                except:
                    st.warning("Please train model first")
                    


else:
    st.title("About Page")


# growth: str = 'linear',
#     changepoints: Any | None = None,
#     n_changepoints: int = 25,
#     changepoint_range: float = 0.8,
#     yearly_seasonality: str = 'auto',
#     weekly_seasonality: str = 'auto',
#     daily_seasonality: str = 'auto',
#     holidays: Any | None = None,
#     seasonality_mode: str = 'additive',
#     seasonality_prior_scale: float = 10,
#     holidays_prior_scale: float = 10,
#     changepoint_prior_scale: float = 0.05,
#     mcmc_samples: int = 0,
#     interval_width: float = 0.8,
#     uncertainty_samples: int = 1000,
#     stan_backend: Any | None = None
# )
