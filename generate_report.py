import pandas as pd

def generate_summary():
    df = pd.read_csv("vehicle_speed_data.csv")

    summary = df.groupby("Vehicle_ID").agg(
        Entry_Frame=("Frame", "min"),
        Exit_Frame=("Frame", "max"),
        Max_Speed=("Speed_kmph", "max"),
        Avg_Speed=("Speed_kmph", "mean")
    ).reset_index()

    summary["Status"] = summary["Max_Speed"].apply(
        lambda x: "Overspeeding" if x > 60 else "Normal"
    )

    summary.to_csv("vehicle_summary_report.csv", index=False)
    print("✅ Summary report saved!")

generate_summary()