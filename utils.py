from datetime import timedelta

    
def yes_no_mapper(choice):
    choice = choice.lower()
    return 1 if choice == 'yes' else 0


def add_days_to_date(user_input_date, model_output_days):
    model_output_days_int = int(model_output_days)
    new_date = user_input_date + timedelta(days=model_output_days_int)

    return new_date.strftime('%Y-%m-%d')


def gender_mapper(choice):
    choice = choice.lower()
    return 1 if choice == 'male' else 0

def category_mapper(choice): 
    choice = choice.lower() 
    if choice in ['low', 'rarely', 'none', 'no drinking/smoking']: 
        return 0 
    elif choice in ['moderate', 'occasionally', 'mild', 'smoking']: 
        return 1 
    elif choice in ['high', 'frequently', 'severe', 'alcohol']: 
        return 2