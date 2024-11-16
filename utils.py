from datetime import timedelta

    
def yes_no_mapper(choice):
    choice = choice.lower()
    return 1 if choice == 'yes' else 0


def add_days_to_date(user_input_date, model_output_days):
    model_output_days_int = int(model_output_days)
    new_date = user_input_date + timedelta(days=model_output_days_int)

    return new_date.strftime('%Y-%m-%d')