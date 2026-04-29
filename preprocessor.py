# import re
# import pandas as pd


# def preprocess(data):

#     pattern = r"\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s?[apAP][mM])?\s-\s"

#     messages = re.split(pattern, data)[1:]
#     dates = re.findall(pattern, data)

#     df = pd.DataFrame({"user_message": messages, "message_date": dates})

#     # CLEAN STEP (mandatory)
#     df["message_date"] = df["message_date"].str.replace(" -", "", regex=False)

#     # PARSE STEP
#     df["message_date"] = pd.to_datetime(
#         df["message_date"], dayfirst=True, errors="coerce"
#     )

#     df.rename(columns={"message_date": "date"}, inplace=True)

#     users = []
#     messages = []
#     for message in df["user_message"]:
#         entry = re.split("([\w\W]+?):\s", message)
#         if entry[1:]:  # user name
#             users.append(entry[1])
#             messages.append(" ".join(entry[2:]))
#         else:
#             users.append("group_notification")
#             messages.append(entry[0])

#     df["user"] = users
#     df["message"] = messages
#     df.drop(columns=["user_message"], inplace=True)

#     df["year"] = df["date"].dt.year
#     df["month_num"] = df["date"].dt.month
#     df["month"] = df["date"].dt.month_name()
#     df["day"] = df["date"].dt.day
#     df["day_name"] = df["date"].dt.day_name()
#     df["hour"] = df["date"].dt.hour
#     df["minute"] = df["date"].dt.minute
#     df["only_date"] = df["date"].dt.date

#     period = []
#     for hour in df[["day_name", "hour"]]["hour"]:
#         if hour == 23:
#             period.append(str(hour) + "-" + str("00"))
#         elif hour == 0:
#             period.append(str("00") + "-" + str(hour + 1))
#         else:
#             period.append(str(hour) + "-" + str(hour + 1))

#     df["period"] = period


#     return df


import re
import pandas as pd


def preprocess(data):

    # ---------------- DATE PARSING ----------------
    pattern = r"\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s?[apAP][mM])?\s-\s"

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({"user_message": messages, "message_date": dates})

    df["message_date"] = df["message_date"].str.replace(" -", "", regex=False)

    df["message_date"] = pd.to_datetime(
        df["message_date"], dayfirst=True, errors="coerce"
    )

    df.rename(columns={"message_date": "date"}, inplace=True)

    # ---------------- USER + MESSAGE ----------------
    users = []
    messages_clean = []

    for message in df["user_message"]:
        entry = re.split("([\w\W]+?):\s", message)

        if entry[1:]:
            users.append(entry[1])
            messages_clean.append(" ".join(entry[2:]))
        else:
            users.append("group_notification")
            messages_clean.append(entry[0])

    df["user"] = users
    df["message"] = messages_clean
    df.drop(columns=["user_message"], inplace=True)

    # ---------------- DATE FEATURES ----------------
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["month"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["day_name"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df["only_date"] = df["date"].dt.date

    # ---------------- PERIOD ----------------
    period = []
    for hour in df["hour"]:
        if hour == 23:
            period.append("23-00")
        elif hour == 0:
            period.append("00-1")
        else:
            period.append(f"{hour}-{hour+1}")

    df["period"] = period

    # =====================================================
    # 🔥 MEDIA LOGIC STARTS HERE
    # =====================================================

    def detect_media(message):
        media_pattern = r"(IMG-\d+-WA\d+\.jpg|VID-\d+-WA\d+\.mp4|AUD-\d+-WA\d+\.opus|PTT-\d+-WA\d+\.opus)"

        if "<Media omitted>" in message:
            return "omitted"

        match = re.search(media_pattern, message)
        if match:
            return match.group()

        return None

    # ✅ extract media filename / omitted
    df["media_file"] = df["message"].apply(detect_media)

    # ✅ flags
    df["is_media"] = df["media_file"].notnull()
    df["is_omitted"] = df["media_file"] == "omitted"
    df["has_actual_file"] = df["media_file"].notnull() & (df["media_file"] != "omitted")

    # ✅ media type (vectorized 🔥)
    df["media_type"] = None

    df.loc[df["media_file"].str.startswith("IMG", na=False), "media_type"] = "image"
    df.loc[df["media_file"].str.startswith("VID", na=False), "media_type"] = "video"
    df.loc[df["media_file"].str.startswith("AUD", na=False), "media_type"] = "audio"
    df.loc[df["media_file"].str.startswith("PTT", na=False), "media_type"] = "voice"

    return df
