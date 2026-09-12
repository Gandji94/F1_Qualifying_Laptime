def quali_time_cleaning():
    import pandas as pd
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sys
    from pathlib import Path
    from .py_def_class import f1_rule_era,abandoned_lap,practice_quali_new,build_session_paths
    import logging

    #we need the dictionary for creating a clear historical order of the different GPs
    f1_calendar_by_season = {
        2018: [
            "Australian GP", "Bahrain GP", "Chinese GP", "Azerbaijan GP",
            "Spanish GP", "Monaco GP", "Canadian GP", "French GP",
            "Austrian GP", "British GP", "German GP", "Hungarian GP",
            "Belgian GP", "Italian GP", "Singapore GP", "Russian GP",
            "Japanese GP", "United States GP", "Mexican GP", "Brazilian GP",
            "Abu Dhabi GP",
        ],

        2019: [
            "Australian GP", "Bahrain GP", "Chinese GP", "Azerbaijan GP",
            "Spanish GP", "Monaco GP", "Canadian GP", "French GP",
            "Austrian GP", "British GP", "German GP", "Hungarian GP",
            "Belgian GP", "Italian GP", "Singapore GP", "Russian GP",
            "Japanese GP", "Mexican GP", "United States GP", "Brazilian GP",
            "Abu Dhabi GP",
        ],

        2020: [
            "Austrian GP", "Styrian GP", "Hungarian GP", "British GP",
            "70th Anniversary GP", "Spanish GP", "Belgian GP", "Italian GP",
            "Tuscan GP", "Russian GP", "Eifel GP", "Portuguese GP",
            "Emilia Romagna GP", "Turkish GP", "Bahrain GP", "Sakhir GP",
            "Abu Dhabi GP",
        ],

        2021: [
            "Bahrain GP", "Emilia Romagna GP", "Portuguese GP", "Spanish GP",
            "Monaco GP", "Azerbaijan GP", "French GP", "Styrian GP",
            "Austrian GP", "British GP", "Hungarian GP", "Belgian GP",
            "Dutch GP", "Italian GP", "Russian GP", "Turkish GP",
            "United States GP", "Mexican GP", "Brazilian GP", "Qatar GP",
            "Saudi Arabian GP", "Abu Dhabi GP",
        ],

        2022: [
            "Bahrain GP", "Saudi Arabian GP", "Australian GP", "Emilia Romagna GP",
            "Miami GP", "Spanish GP", "Monaco GP", "Azerbaijan GP",
            "Canadian GP", "British GP", "Austrian GP", "French GP",
            "Hungarian GP", "Belgian GP", "Dutch GP", "Italian GP",
            "Singapore GP", "Japanese GP", "United States GP", "Mexican GP",
            "Brazilian GP", "Abu Dhabi GP",
        ],

        2023: [
            "Bahrain GP", "Saudi Arabian GP", "Australian GP", "Azerbaijan GP",
            "Miami GP", "Monaco GP", "Spanish GP", "Canadian GP",
            "Austrian GP", "British GP", "Hungarian GP", "Belgian GP",
            "Dutch GP", "Italian GP", "Singapore GP", "Japanese GP",
            "Qatar GP", "United States GP", "Mexican GP", "Brazilian GP",
            "Las Vegas GP", "Abu Dhabi GP",
        ],

        2024: [
            "Bahrain GP", "Saudi Arabian GP", "Australian GP", "Japanese GP",
            "Chinese GP", "Miami GP", "Emilia Romagna GP", "Monaco GP",
            "Canadian GP", "Spanish GP", "Austrian GP", "British GP",
            "Hungarian GP", "Belgian GP", "Dutch GP", "Italian GP",
            "Azerbaijan GP", "Singapore GP", "United States GP", "Mexican GP",
            "Brazilian GP", "Las Vegas GP", "Qatar GP", "Abu Dhabi GP",
        ],

        2025: [
            "Australian GP", "Chinese GP", "Japanese GP", "Bahrain GP",
            "Saudi Arabian GP", "Miami GP", "Emilia Romagna GP", "Monaco GP",
            "Spanish GP", "Canadian GP", "Austrian GP", "British GP",
            "Belgian GP", "Hungarian GP", "Dutch GP", "Italian GP",
            "Azerbaijan GP", "Singapore GP", "United States GP", "Mexican GP",
            "Brazilian GP", "Las Vegas GP", "Qatar GP", "Abu Dhabi GP",
        ],

        #2026 was already adjusted due to the canceld Saudi Arabia and Barhain GPs
        2026: [
            "Australian GP", "Chinese GP", "Japanese GP", "Miami GP",
            "Canadian GP", "Monaco GP", "Spanish GP", "Austrian GP",
            "British GP", "Belgian GP", "Hungarian GP", "Dutch GP",
            "Italian GP", "Madrid GP", "Azerbaijan GP", "Singapore GP",
            "United States GP", "Mexican GP", "Brazilian GP", "Las Vegas GP",
            "Qatar GP", "Abu Dhabi GP",
        ],
    }

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info("Setting up data paths")
    try:
        ROOT = Path(__file__).resolve().parent.parent
        DATA = ROOT /"data"
        DATA_RAW = DATA / "raw"
        DATA_OUTPUT = DATA/"processed"
        DATA_DBT_OUTPUT = ROOT/'f1_qualifying_dbt'/'seeds'
    except Exception as ex:
        logging.error(f'Creating the paths failed: {ex}')
        raise

    logging.info('Creating F1 circut data set')
    try:
        input_circut = DATA_RAW / "f1_circuits_2018_2026_extended.xlsx"
        df_circut=pd.read_excel(input_circut).iloc[:,:-4]
        df_circut['Circut_length'] = df_circut['Circut_length'].apply(lambda x: round(float(x.split(' ')[0]),2))
        df_circut['Pace_profile'] = df_circut['Pace_profile'].apply(lambda x: x.split('/')[0] if x.split('/') else x)
        df_circut.drop(['Confidence'],axis=1,inplace=True)
        #adding 
        df_circut_adjust = df_circut[df_circut['Location'].isin(['Silverstone','Sakhir','Spielberg'])].replace({
            'Bahrain Grand Prix; Sakhir Grand Prix':'Sakhir Grand Prix',
            'Austrian Grand Prix; Styrian Grand Prix':'Styrian Grand Prix',
            'British Grand Prix; 70th Anniversary Grand Prix':'70th Anniversary Grand Prix'
            }).copy()
        df_circut = pd.concat([df_circut,df_circut_adjust],axis=0)
        df_circut.rename(columns={'Grand_Prix(es)':'GP'},inplace=True)
        df_circut.replace({
            'Bahrain Grand Prix; Sakhir Grand Prix':'Bahrain Grand Prix',
            'Austrian Grand Prix; Styrian Grand Prix':'Austrian Grand Prix',
            'British Grand Prix; 70th Anniversary Grand Prix':'British Grand Prix',
            'Italian Grand Prix; San Marino Grand Prix; Emilia-Romagna Grand Prix':'Emilia Romagna Grand Prix',
            'Brazilian Grand Prix; São Paulo Grand Prix':'Brazilian Grand Prix',
            'Mexican Grand Prix; Mexico City Grand Prix':'Mexican Grand Prix',
            'European Grand Prix; Azerbaijan Grand Prix':'Azerbaijan Grand Prix',
            'Spanish Grand Prix':'Madrid Grand Prix',
            'Spanish Grand Prix; Barcelona-Catalunya Grand Prix':'Spanish Grand Prix',
            'German Grand Prix; European Grand Prix; Luxembourg Grand Prix; Eifel Grand Prix':'Eifel Grand Prix',
            },inplace=True)
        df_circut['GP'] = df_circut['GP'].apply(lambda x: x.replace('Grand Prix','GP'))
        df_circut = df_circut[['GP','Type','Direction','Circut_length','Turns','Pace_profile',
                'Flat_out_run','Slow_turns','Medium_turns','High_speed_turns','Turn_density','Complexity_label']]
        output_f1_circut = DATA_OUTPUT/"cleaned_f1_circut.csv"
        df_circut.to_csv(output_f1_circut,index=False)
    except Exception as ex:
        logging.error(f'Creating the F1 circut data set failed: {ex}')
        raise

    logging.info('Creating qualifying data set')
    try:
        df_2018_2026 = pd.DataFrame()

        for season in range(2018, 2027):
            #turning "season" to string, because the path cannot processes a numerical value
            data_path = DATA_RAW / str(season)

            #getting name of folder if the folder exists
            path_list = sorted([f.name for f in data_path.iterdir() if f.is_dir() and not f.name.startswith('.') and f.name.endswith("GP")])

            # keep this only if you really need it
            #if season <= 2026:
            #path_list = path_list[1:]

            if season == 2020:
                path_list = [x for x in path_list if x != "Eifel GP"]

            #entering each folder of a Grand Prix weekend
            for folder_loop_value in path_list:
                base = data_path / folder_loop_value

                timing_paths, weather_paths = build_session_paths(
                    base=base,
                    season=season,
                    folder_loop_value=folder_loop_value
                )

                df_loop = practice_quali_new(
                    season=season,
                    gp=folder_loop_value,
                    timing_paths=timing_paths,
                    weather_paths=weather_paths,
                    w_vis=False
                )

                df_2018_2026 = pd.concat([df_2018_2026, df_loop], ignore_index=True)
        
        #creating the mapping => key: season, gp; value: number of the gp-weekend
        race_round_map = {
            (season, gp): round_no
            for season, races in f1_calendar_by_season.items()
            for round_no, gp in enumerate(races, start=1)
        }

        #race_round will be used as an X-feature
        df_2018_2026["race_round"] = df_2018_2026.apply(
            lambda row: race_round_map.get((row["Season"], row["GP"])),
            axis=1
        )

        #creating a dataframe to create an race order
        weekend_order = (
            df_2018_2026[["Season", "GP", "race_round"]]
            .drop_duplicates()
            .sort_values(["Season", "race_round"])
            .reset_index(drop=True)
        )
        #creating a unique id based on the race weekend order
        #group_id will be used for time-aware cross-validation
        weekend_order["gp_id"] = range(len(weekend_order))

        #finally joining the race order logic
        df_2018_2026 = df_2018_2026.merge(
            weekend_order[["Season", "GP", "gp_id"]],
            on=["Season", "GP"],
            how="left"
        )

        team_lineage_map = {
            "Toro Rosso": "Toro Rosso-AlphaTauri-RB-Racing Bulls",
            "AlphaTauri": "Toro Rosso-AlphaTauri-RB-Racing Bulls",
            "RB": "Toro Rosso-AlphaTauri-RB-Racing Bulls",
            "Racing Bulls": "Toro Rosso-AlphaTauri-RB-Racing Bulls",

            "Sauber": "Sauber-Alfa Romeo-Kick Sauber-Audi",
            "Alfa Romeo": "Sauber-Alfa Romeo-Kick Sauber-Audi",
            "Alfa Romeo Racing": "Sauber-Alfa Romeo-Kick Sauber-Audi",
            "Kick Sauber": "Sauber-Alfa Romeo-Kick Sauber-Audi",
            "Audi": "Sauber-Alfa Romeo-Kick Sauber-Audi",

            "Force India":"Force India-Racing Point-Aston Martin",
            "Racing Point":"Force India-Racing Point-Aston Martin",
            "Aston Martin":"Force India-Racing Point-Aston Martin",
            "Alpine":"Alpine-Renault",
            "Renault":"Alpine-Renault"
        }

        df_2018_2026["Team_Lineage"] = df_2018_2026["Team"].replace(team_lineage_map)

        #output_file = DATA_OUTPUT / "df_2018_2026_quali_output.csv"
        #df_2018_2026.to_csv(output_file, index=False)
    except Exception as ex:
        logging.error(f'Creating the data set failed: {ex}')
        raise

    logging.info("Combining quali data set with the f1 circut data set")
    try:
        df_quali = pd.merge(df_2018_2026,df_circut,how='left',on='GP')
        df_quali["Pace_profile"] = df_quali["Pace_profile"].str.strip()
        #in case, we want to create the data frame withotu the Nan values already
        #df_quali = df_quali[df_quali['laptime_sum_sectortimes_quali'].notna()].copy()
        df_q_date = pd.read_csv(DATA_RAW/'f1_gp_qualifying_dates_2018_2026.csv')
        df_quali = df_quali.merge(df_q_date,on=['Season','GP'],how='inner').copy()
        final_output = DATA_OUTPUT/"Cleaned_quali_f1.csv"
        dbt_final_output = DATA_DBT_OUTPUT/"Cleaned_quali_f1.csv"
        
        df_quali.to_csv(final_output,index=False)
        df_quali.to_csv(dbt_final_output,index=False)
    except Exception as ex:
        logging.error(f'Combining quali data set with the f1 circut data set failed: {ex}')
        raise

    logging.info('Data set is created')

if __name__ == "__main__":
    quali_time_cleaning()