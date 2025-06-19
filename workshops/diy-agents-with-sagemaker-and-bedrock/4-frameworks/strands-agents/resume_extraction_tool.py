from strands.models.bedrock import BedrockModel
from strands import Agent, tool
import boto3
# Amazon Textract Client
textract = boto3.client('textract', region_name='us-west-2')
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from a resume image using Amazon Textract.
    """
    with open(image_path, 'rb') as document:
        image_bytes = document.read()
    response = textract.detect_document_text(Document={'Bytes': image_bytes})
    lines = [item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE']
    return '\n'.join(lines)
# Define Bedrock model (lightweight for post-processing if needed)
resume_model = BedrockModel(model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0")
@tool
def resume_extraction_assistant(image_path: str) -> str:
    """
    Tool to extract raw text from a resume image using Textract.
    """
    resume_text = extract_text_from_image(image_path)
    
    # Optional: Use Claude to summarize/extract key insights
    summarizer = Agent(
        model=resume_model,
        system_prompt="""
        You are a resume parsing assistant. Extract and summarize key details from raw resume text, 
        including name, email, phone, skills, experience, education, and certifications. 
        Output in clean bullet points or JSON format.
        """
    )
    
    return summarizer(resume_text).message