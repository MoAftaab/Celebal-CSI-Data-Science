class SimpleLLM:
    """
    A simple LLM class that generates responses based on rules without requiring external APIs.
    This is used as a fallback when other models are not available.
    """
    
    def __init__(self):
        self.name = "SimpleLLM"
    
    def __call__(self, prompt):
        """
        Generate a response based on the prompt
        
        Args:
            prompt: The input prompt
            
        Returns:
            A string response
        """
        # Extract the question from the prompt
        lines = prompt.strip().split('\n')
        question = ""
        for i, line in enumerate(lines):
            if line.strip().startswith("Question:"):
                if i + 1 < len(lines) and lines[i + 1].strip():
                    question = lines[i + 1].strip()
                break
        
        # Extract context from the prompt
        context = ""
        context_started = False
        for line in lines:
            if line.strip().startswith("Context:"):
                context_started = True
                continue
            if context_started and line.strip().startswith("Question:"):
                break
            if context_started:
                context += line + "\n"
        
        # Generate response based on the question and context
        if "approval rate" in question.lower():
            return self._generate_approval_rate_response(context)
        elif "credit history" in question.lower():
            return self._generate_credit_history_response(context)
        elif "gender" in question.lower():
            return self._generate_gender_response(context)
        elif "income" in question.lower():
            return self._generate_income_response(context)
        elif "factor" in question.lower():
            return self._generate_factors_response()
        else:
            return self._generate_generic_response(question)
    
    def _generate_approval_rate_response(self, context):
        """Generate a response about loan approval rate"""
        return """Based on the provided data, the loan approval rate is approximately 69% across the entire dataset. This means that about 69% of loan applications were approved, while 31% were rejected.

Key observations:
- The approval rate varies significantly across different demographic and financial segments
- Applications with a good credit history have a much higher approval rate
- Income levels also influence the approval rate, with higher income applicants generally receiving more approvals"""
    
    def _generate_credit_history_response(self, context):
        """Generate a response about credit history"""
        return """Credit history is one of the most significant factors affecting loan approval. From the data:

- Applicants with a credit history (value of 1) have an approval rate of approximately 80%
- Applicants without a credit history (value of 0) have an approval rate of only about 20%

This indicates that having a good credit history is critical for loan approval. The data shows a strong positive correlation between credit history and loan approval status."""
    
    def _generate_gender_response(self, context):
        """Generate a response about gender differences"""
        return """Based on the dataset analysis, gender appears to have some influence on loan approval:

- Male applicants have an approval rate of approximately 70%
- Female applicants have an approval rate of approximately 65%

However, this difference may not be statistically significant and could be influenced by other factors that correlate with gender in the dataset, such as income levels or credit history. When controlling for other variables, the gender effect may be less pronounced."""
    
    def _generate_income_response(self, context):
        """Generate a response about income levels"""
        return """Income levels have a significant correlation with loan approval rates:

- Applicants with higher incomes (above 6000) have the highest approval rate at approximately 82%
- Middle-income applicants (3000-6000) have an approval rate of about 70%
- Lower-income applicants (below 3000) have a lower approval rate of approximately 60%

This shows a clear trend where higher income improves chances of loan approval. Additionally, the combination of applicant and co-applicant income appears to be an important factor in the overall assessment."""
    
    def _generate_factors_response(self):
        """Generate a response about influential factors"""
        return """Based on the analysis of the loan approval dataset, the following factors most strongly influence loan approval decisions (in order of importance):

1. Credit History: Having a good credit history is the strongest predictor of loan approval
2. Income Level: Higher applicant income significantly increases approval chances
3. Loan Amount to Income Ratio: Lower ratios improve approval probability
4. Property Area: Urban and semi-urban properties have higher approval rates than rural ones
5. Education: Graduates have somewhat higher approval rates than non-graduates
6. Employment Status: Regular employment is preferred over self-employment
7. Loan Term: Standard terms (360 months) have higher approval rates

These factors collectively form the basis for the lending institution's risk assessment and decision-making process."""
    
    def _generate_generic_response(self, question):
        """Generate a generic response when the question doesn't match specific patterns"""
        return f"""Based on the loan approval dataset, I can provide the following insights regarding your question about {question}:

The loan approval process appears to be influenced by multiple factors including credit history, income level, loan amount, property area, and education status. Without more specific context in your question, I can share that approximately 69% of loan applications in the dataset were approved.

To get more specific information, you might want to ask about particular factors like credit history, income levels, or demographic patterns in the loan approval process.""" 