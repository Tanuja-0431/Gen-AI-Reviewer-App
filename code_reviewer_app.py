import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Hugging Face model and tokenizer
model_name = "gpt2"  # Change this if needed
token="YOUR_GOOGLE_API"
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./cache")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache")
tokenizer.pad_token = tokenizer.eos_token

# Streamlit UI
st.title("🧑‍💻 GenAI Code Reviewer")
st.write("Submit your Python code for review and receive feedback along with fixed snippets. 🚀")

# Input Section
code_input = st.text_area("📝 Enter your Python code here:", height=200)

# Function to analyze Python code
def analyze_code_with_huggingface(code):
    # Basic syntax check to handle errors
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        st.error(f"❌ Syntax Error: {e}")
        corrected_code = suggest_correction(code, e)
        return None, corrected_code  # Return None if syntax is invalid, along with corrected code
    
    # Prepare the input prompt for the model
    prompt = f"""
    You are an expert Python code reviewer. Analyze the following code and provide:
    1. A list of issues or bugs, including logical errors.
    2. Suggestions for improvement.
    3. A fixed version of the code if there are errors.

    Please format your response clearly, for example:
    - **Issues:** [list of issues]
    - **Suggestions:** [list of suggestions]
    - **Fixed Code:** [fixed code snippet]

    If the code is correct, please confirm that it looks good and suggest any best practices.

    Code:
    {code}
    """
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate model output
    try:
        outputs = model.generate(
            inputs['input_ids'],
            max_length=500,  # Limit the length of the response
            num_return_sequences=1,
            num_beams=5,
            repetition_penalty=1.2,
            early_stopping=True
        )
        
        # Decode and return the result
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result, None  # Return the result and None for corrected code
    except Exception as e:
        st.error(f"⚠️ Error during model inference: {e}")
        return None, None

# Function to suggest corrections for common syntax errors
def suggest_correction(code, error):
    error_message = str(error)
    
    # Example corrections for common syntax errors
    if "unexpected EOF while parsing" in error_message:
        return code + "\n# Added missing closing bracket or parenthesis"
    elif "invalid syntax" in error_message:
        # Attempt to fix common issues
        lines = code.splitlines()
        for i, line in enumerate(lines):
            if line.strip().endswith(":") and not line.strip().endswith(":\n"):
                lines[i] += " # Added missing indentation or code block"
            elif line.strip() == "":
                lines[i] = "# Check for missing code here"
        return "\n".join(lines)  # Return the modified code
    # Add more specific error handling as needed

    return code  # Return the original code if no specific correction is suggested

# Function to parse response into issues, suggestions, and fixed code
def parse_response(response):
    issues = "✅ No issues found."
    suggestions = "💡 No suggestions provided."
    fixed_code = "🔧 No fixed code provided."

    # Improved parsing logic
    if "Issues:" in response:
        issues = response.split("Issues:")[1].split("Suggestions:")[0].strip()
    if "Suggestions:" in response:
        suggestions = response.split("Suggestions:")[1].split("Fixed Code:")[0].strip()
    if "Fixed Code:" in response:
        fixed_code = response.split("Fixed Code:")[1].strip()
    
    # Check if the code is correct
    if "looks good" in response .lower():
        issues = "✅ No issues found."
        suggestions = "The code is correct. No changes needed. 🎉"
        fixed_code = "🔧 No fixed code provided."

    return issues, suggestions, fixed_code

# Analyze the code when the button is pressed
if st.button("🔍 Submit for Review"):
    if code_input.strip():
        st.info("🔄 Analyzing your code...")
        # Call the function to analyze the code
        response, corrected_code = analyze_code_with_huggingface(code_input)

        if response:
            issues, suggestions, fixed_code = parse_response(response)
            st.success("✅ Code Review Complete!")
            
            # Display results
            st.subheader("🛠️ Issues:")
            st.write(issues)
            
            st.subheader("💡 Suggestions:")
            st.write(suggestions)
            
            st.subheader("🔧 Suggested Fix:")
            st.code(fixed_code, language='python')
        
        if corrected_code:
            st.subheader("🛠️ Suggested Correction for Syntax Error:")
            st.code(corrected_code, language='python')
    else:
        st.warning("⚠️ Please enter some code before submitting.")