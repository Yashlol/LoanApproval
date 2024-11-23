import joblib
from django.conf import settings
from django.shortcuts import render
from .forms import LoanApprovalForm
from django.core.exceptions import ObjectDoesNotExist
prediction="Not Approved"
def loan_approval_view(request):
    """
    Handle loan approval form submission and prediction using a pre-trained model.
    """
    if request.method == 'POST':
        form = LoanApprovalForm(request.POST)
        if form.is_valid():
            # Extract form data
            try:
                data = form.cleaned_data  # Already in the correct format
                feature_order = [
                    'no_of_dependents', 'education', 'self_employed', 'income_annum',
                    'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
                    'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
                ]

                # Ensure input matches the feature order
                input_data = [float(data[feature]) for feature in feature_order]

                # Load the model and label encoders
                model = joblib.load(settings.PREDICTOR_CONFIG['model'])

                # Perform prediction
                result = model.predict([input_data])[0]
                prediction = "Approved" if result == 1 else "Not Approved"
                return render(request, 'result.html', {'result': prediction})
            
            except FileNotFoundError:
                # Handle missing model file
                error_message = "The prediction model file is missing. Please check the configuration."
                return render(request, 'error.html', {'error_message': error_message})
            
            except ObjectDoesNotExist:
                # Handle configuration issues
                error_message = "The model configuration is invalid or missing."
                return render(request, 'error.html', {'error_message': error_message})
            
            except ValueError as ve:
                # Handle issues with input data
                error_message = f"Invalid input data: {ve}"
                return render(request, 'error.html', {'error_message': error_message})
            
            except Exception as e:
                # Catch-all for unexpected errors
                error_message = f"An unexpected error occurred: {str(e)}"
                return render(request, 'error.html', {'error_message': error_message})
    else:
        # Render the form for a GET request
        form = LoanApprovalForm()

    return render(request, 'form.html', {'form': form})

def result(request):
    return render(request, 'result.html', {'result': prediction})

def home(request):
    """
    Render the home page.
    """
    return render(request, 'home.html')  # Ensure you have a 'home.html' template
