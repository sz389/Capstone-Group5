def manual_label_encoder(df_label, category = "emotion"):
    label_list = []
    if category == "parkinson":
        for label in df_label:
            if label == 'hc':
                label_list.append(0)
            else:
                label_list.append(1)

    elif category == "age":
        for label in df_label:
            if label == "<30s":
                label_list.append(0)
            elif label == "30s":
                label_list.append(1)
            elif label == "40s":
                label_list.append(2)
            elif label == "50s":
                label_list.append(3)
            elif label == ">60s":
                label_list.append(4)

    elif category == "race":
        for label in df_label:
            if label == "Caucasian":
                label_list.append(0)
            elif label == "African American":
                label_list.append(1)
            elif label == "Asian":
                label_list.append(2)

    elif category == 'sex':
        for label in df_label:
            if label == "Male":
                label_list.append(0)
            else:
                label_list.append(1)

    elif category == "emotion":
        for label in df_label:
            if label == "ANG":
                label_list.append(0)
            elif label == 'DIS':
                label_list.append(1)
            elif label == "FEA":
                label_list.append(2)
            elif label == "HAP":
                label_list.append(3)
            elif label == "NEU":
                label_list.append(4)
            elif label == "SAD":
                label_list.append(5)

    elif category == "accent":
        for label in df_label:
            if label == "arabic":
                label_list.append(0)
            elif label == "english":
                label_list.append(1)
            elif label == "french":
                label_list.append(2)
            elif label == "mandarin":
                label_list.append(3)
            else:
                label_list.append(4)

    return label_list
