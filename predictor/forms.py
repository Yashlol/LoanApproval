from django import forms

class LoanApprovalForm(forms.Form):
    no_of_dependents = forms.IntegerField(
        label="Number of Dependents", 
        help_text="Enter the number of dependents you have. This includes children, spouse, or anyone financially dependent on you."
    )
    education = forms.ChoiceField(
        choices=[(1, 'Graduate'), (0, 'Not Graduate')],
        label="Education",
        help_text="Select your highest level of education. Graduate means you have completed a bachelor's degree or higher."
    )
    self_employed = forms.ChoiceField(
        choices=[(1, 'Yes'), (0, 'No')],
        label="Self Employed",
        help_text="Select 'Yes' if you are self-employed, or 'No' if you are employed by someone else."
    )
    income_annum = forms.IntegerField(
        label="Annual Income",
        help_text="Enter your total income for the year before taxes. This includes salary, business income, etc."
    )
    loan_amount = forms.IntegerField(
        label="Loan Amount",
        help_text="Enter the amount of loan you wish to apply for."
    )
    loan_term = forms.IntegerField(
        label="Loan Term",
        help_text="Enter the term of the loan in years. For example, 15 years."
    )
    cibil_score = forms.IntegerField(
        label="CIBIL Score",
        help_text="This is your credit score. It reflects your creditworthiness and repayment history. A higher score means you're more likely to get approved for the loan."
    )
    residential_assets_value = forms.IntegerField(
        label="Residential Asset Value",
        help_text="Enter the value of your residential property or home."
    )
    commercial_assets_value = forms.IntegerField(
        label="Commercial Asset Value",
        help_text="Enter the value of any commercial properties you own, such as office buildings or rental properties."
    )
    luxury_assets_value = forms.IntegerField(
        label="Luxury Asset Value",
        help_text="Enter the value of any luxury assets, such as cars, jewelry, or other high-value items."
    )
    bank_asset_value = forms.IntegerField(
        label="Bank Asset Value",
        help_text="Enter the value of assets you have in the bank, such as savings, fixed deposits, or other investments."
    )
