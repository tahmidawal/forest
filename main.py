from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
from pytesseract import image_to_string
import json
import boto3
import os
from tempfile import NamedTemporaryFile
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import boto3
import json
import re
from io import BytesIO
import pandas as pd

s3_client = boto3.client('s3')
bucket_name = 'forestai'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataPoints(BaseModel):
    data_points: str

def convert_pdf_to_images(file_path, scale=300/72):
    pdf_file = pdfium.PdfDocument(file_path)
    page_indices = [i for i in range(len(pdf_file))]
    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )
    final_images = []
    for i, image in zip(page_indices, renderer):
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append({i: image_byte_array})
    return final_images

def extract_text_from_img(list_dict_final_images):
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []
    for image_bytes in image_list:
        image = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(image))
        image_content.append(raw_text)
    return "\n".join(image_content)

# def extract_content_from_file(file: UploadFile):
#     temp_file = NamedTemporaryFile(delete=False)
#     temp_file.write(file.file.read())
#     temp_file.close()

#     images_list = convert_pdf_to_images(temp_file.name)
#     os.remove(temp_file.name)

#     text_with_pytesseract = extract_text_from_img(images_list)

#     return text_with_pytesseract

def extract_content_from_file(file_content):
    images_list = convert_pdf_to_images(BytesIO(file_content))
    text_with_pytesseract = extract_text_from_img(images_list)
    return text_with_pytesseract

def extract_structured_data_mistral(content: str, data_points: list):
    bedrock_runtime_client = boto3.client(service_name="bedrock-runtime")
    
    # Build the template to instruct the model
    template = f"""
    You are an expert admin tasked with extracting transaction information from financial documents.
    {content}
    Above is the content; please try to extract all data points from the content above 
    and export in a JSON array format:
    {data_points}
    Now please extract details from the content and export in a JSON array format, 
    return ONLY the JSON array and nothing else.
    """
    
    # Wrap the template with instruct tags for Mistral
    instruction = f"<s>[INST] {template} [/INST]"
    
    # Configuration for the Mistral 8x7B model
    body = {
        "prompt": instruction,
        "max_tokens": 4000,  # Adjusted max_tokens to 2000 based on the amount of content
        "temperature": 0.0,  # Adjust the temperature as needed
    }

    # Invoke the Mistral model
    response = bedrock_runtime_client.invoke_model(
        modelId="mistral.mixtral-8x7b-instruct-v0:1",
        body=json.dumps(body)
    )
    print(response)

    response_body = json.loads(response["body"].read())
    outputs = response_body.get("outputs")

    # Extracting the text completions from the outputs
    completions = [output["text"] for output in outputs]
    return completions



# 3. Extract structured info from text via LLM
def extract_structured_data_llama(content: str, data_points):

    bedrock=boto3.client(service_name="bedrock-runtime")
    bedrock_embeddings=BedrockEmbeddings(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock)

    template = f"""
    You are an expert admin tasked with extracting transaction information from financial documents.

    {content}

    Above is the content; please try to extract all data points from the content above 
    and export in a JSON array format:
    {data_points}

    Now please extract details from the content  and export in a JSON array format, 
    return ONLY and only the JSON array and nothing else.
    """

    # prompt = PromptTemplate(
    #     input_variables=["content", "data_points"],
    #     template=template,
    # )
    # Turn prompt into a string
    #body = json.dumps(prompt)
    
    payload = {
        "prompt": "[INST]" + template + "[/INST]",
        "max_gen_len": 512,
        "temperature": 0.0,
        "top_p": 0.9
    }
    
    body = json.dumps(payload)

    response = bedrock.invoke_model(
        body=body,
        modelId = "meta.llama2-70b-chat-v1",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get('body').read())

    return response_body

def extract_structured_data_claude( prompt):
    """
    Invokes Anthropic Claude 3 Sonnet to run an inference using the input
    provided in the request body.

    :param prompt: The prompt that you want Claude 3 to complete.
    :return: Inference response from the model.
    """

    # Initialize the Amazon Bedrock runtime client
    bedrock=boto3.client(service_name="bedrock-runtime")

    # Invoke Claude 3 with the text prompt
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
            }
        ),
    )

    # Process and print the response
    result = json.loads(response.get("body").read())
    input_tokens = result["usage"]["input_tokens"]
    output_tokens = result["usage"]["output_tokens"]
    output_list = result.get("content", [])
    # convert the output list to one string
    output_string = "\n".join([output["text"] for output in output_list])

    # print("Invocation details:")
    # print(f"- The input length is {input_tokens} tokens.")
    # print(f"- The output length is {output_tokens} tokens.")

    #print(f"- The model returned {len(output_list)} response(s):")
    # for output in output_list:
    #     print(output["text"])

    return output_string

# data_points = """
# {
#     "Cash": "Received money of proceeds from the distribution of the estate or foundation",
#     "Cost basis": "The cost basis of the proceeds",
#     "Long-term capital gain/(loss)": "Long-term capital gain/(loss)",
#     "Interest income": "Interest income",
#     "Dividend income": "Dividend income",
#     "Amount withheld from distribution": "Amount withheld from this distribution (if any)",
#     "Net Distribution Proceedings": "Net distribution proceeds of the estate or foundation"
# }
# """
data_points = """
{
    "Customer Id": "Identifier of the customer. For consistency across transactions, use a predefined or default identifier where not specified.",
    "Entity": "Name of the entity receiving the distribution or involved in the transaction.",
    "Custodian": "Optional. The custodian of the account, if different from the entity.",
    "Account Number/Name": "The specific account or fund identifier involved in the transaction. If not specified, say "None".",
    "Trans Type": "Type of transaction, typically 'CREDIT' for distributions or 'DEBIT' for charges.",
    "Security Type": "Type of security involved, generally 'CASH' for cash transactions.",
    "Symbol": "Optional. The trading symbol for a security, if applicable.",
    "Trade Date": "Date of transaction initiation in YYYYMMDD format.",
    "Settlement Date": "Date of transaction completion in YYYYMMDD format.",
    "Units": "Optional. The number of units involved in the transaction, if none given then assume USD.",
    "Amount": "The value of the net distribution proceed for the customer",
    "Currency Code": "Currency of the transaction, usually 'USD'.",
    "Name": "Descriptive name of the transaction or operation.",
    "Description": "Description of the transaction or operation.",
    "Breakdown": "Detailed breakdown description of the components of this distribution.",
    "Check Number": "If the transaction involves a check, include the check number.",
    "Tran Sub-Type": " Additional categorization detail of the transaction type if available.",
    "Accrued Interest": "Optional. Any interest that has accrued relevant to the transaction.",
    "LTGL": "The value of Long Term Gain/Loss in content. If not mentioned, use '0'.",
    "Original Sec Type": " The original type of security if changed during the transaction process.",
    "Original Tran Type": " The original transaction type if it was modified in the process.",
    "Transaction ID": " A unique identifier for tracking the transaction."
    "STGL": "The value of Short Term Gain/Loss in content. If not mentioned, use '0'.",
}
"""
data_points_capital_call_1 = """
{
    "Customer Id": "Identifier of the customer. For consistency across transactions, use a predefined or default identifier where not specified.",
    "Entity": "Name of the entity receiving the distribution or involved in the transaction.",
    "Custodian": "Optional. The custodian of the account, if different from the entity.",
    "Account Number/Name": "The specific account or fund identifier involved in the transaction. If not specified, say "None".",
    "Trans Type": "Type of transaction, typically 'CREDIT' for distributions or 'DEBIT' for charges.",
    "Security Type": "Type of security involved, generally 'CASH' for cash transactions.",
    "Symbol": "Optional. The trading symbol for a security, if applicable.",
    "Trade Date": "Date of transaction initiation in YYYYMMDD format.",
    "Settlement Date": "Date of transaction completion in YYYYMMDD format.",
    "Units": "Optional. The number of units involved in the transaction, if none given then assume USD.",
    "Amount": "The value of the capital call amount for the customer ABC31 I11-13",
    "Currency Code": "Currency of the transaction, usually 'USD'.",
    "Name": "Descriptive name of the transaction or operation.",
    "Description": "Description of the transaction or operation.",
    "Breakdown": "Detailed breakdown description of the components of this distribution.",
    "Check Number": "If the transaction involves a check, include the check number.",
    "Tran Sub-Type": " Additional categorization detail of the transaction type if available.",
    "Accrued Interest": "Optional. Any interest that has accrued relevant to the transaction.",
    "LTGL": "The value of Long Term Gain/Loss in content. If not mentioned, use '0'.",
    "Original Sec Type": " The original type of security if changed during the transaction process.",
    "Original Tran Type": " The original transaction type if it was modified in the process.",
    "Transaction ID": " A unique identifier for tracking the transaction."
    "STGL": "The value of Short Term Gain/Loss in content. If not mentioned, use '0'.",
}
"""
data_points_capital_call = """
{
    "Customer Id": "Identifier of the customer. For consistency across transactions, use a predefined or default identifier where not specified.",
    "Entity": "Name of the entity receiving the distribution or involved in the transaction.",
    "Custodian": "Optional. The custodian of the account, if different from the entity.",
    "Account Number/Name": "The specific account or fund identifier involved in the transaction. If not specified, say "None".",
    "Trans Type": "Type of transaction, typically 'CREDIT' for distributions or 'DEBIT' for charges.",
    "Security Type": "Type of security involved, generally 'CASH' for cash transactions.",
    "Symbol": "Optional. The trading symbol for a security, if applicable.",
    "Trade Date": "Date of transaction initiation in YYYYMMDD format.",
    "Settlement Date": "Date of transaction completion in YYYYMMDD format.",
    "Units": "Optional. The number of units involved in the transaction, if none given then assume USD.",
    "Amount": "The value of the capital call amount for the customer",
    "Currency Code": "Currency of the transaction, usually 'USD'.",
    "Name": "Descriptive name of the transaction or operation.",
    "Description": "Description of the transaction or operation.",
    "Breakdown": "Detailed breakdown description of the components of this distribution.",
    "Check Number": "If the transaction involves a check, include the check number.",
    "Tran Sub-Type": " Additional categorization detail of the transaction type if available.",
    "Accrued Interest": "Optional. Any interest that has accrued relevant to the transaction.",
    "LTGL": "The value of Long Term Gain/Loss in content. If not mentioned, use '0'.",
    "Original Sec Type": " The original type of security if changed during the transaction process.",
    "Original Tran Type": " The original transaction type if it was modified in the process.",
    "Transaction ID": " A unique identifier for tracking the transaction."
    "STGL": "The value of Short Term Gain/Loss in content. If not mentioned, use '0'.",
}
"""
data_points2 = """
{
    "Currency Code": "Currency of the transaction, usually 'USD'.",
    "Name": "Descriptive name of the transaction or operation.",
    "Description": "Description of the transaction or operation.",
    "Breakdown": "Detailed breakdown of the components of this distribution in pargraph format.",
    "Check Number": "If the transaction involves a check, include the check number.",
    "Tran Sub-Type": " Additional categorization detail of the transaction type if available.",
    "Accrued Interest": "Optional. Any interest that has accrued relevant to the transaction.",
    "STGL": "Short Term Gain/Loss ",
    "LTGL": "Long Term Gain/Loss ",
    "Original Sec Type": " The original type of security if changed during the transaction process.",
    "Original Tran Type": " The original transaction type if it was modified in the process.",
    "Transaction ID": " A unique identifier for tracking the transaction."
}
"""


bucket_name = "forestai"
@app.post("/extract_PE")
async def extract_dataPE( file_key: str):
    try:
        # Fetch the file from S3
        file_key2 = file_key + ".pdf"
        s3_response = s3_client.get_object(Bucket=bucket_name, Key=file_key2)
        file_content = s3_response['Body'].read()

        # Process the file content
        content = extract_content_from_file(file_content)
        print(content)
        extracted_info = extract_structured_data_llama(content, data_points)
        print(extracted_info.get("generation"))

        json_string = re.search(r'\{[^}]+\}', extracted_info.get("generation"), re.DOTALL).group()
        data = json.loads(json_string)
        
        print(data)
        
        # Convert the data to xlsx
        df = pd.DataFrame([data])
        df.to_excel("output.xlsx", index=False)
        # Upload the xlsx file to S3 named after the input file
        s3_client.upload_file("output.xlsx", bucket_name, f"{file_key}.xlsx")
        
        return JSONResponse(content=data)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/extract_Capital_Call1")
async def extract_data_CapCall1( file_key: str):
    try:
        # Fetch the file from S3
        
        file_key2 = file_key + ".pdf"

        s3_response = s3_client.get_object(Bucket=bucket_name, Key=file_key2)
        file_content = s3_response['Body'].read()

        # Process the file content
        content = extract_content_from_file(file_content)
        print(content)
        extracted_info = extract_structured_data_llama(content, data_points_capital_call_1)
        print(extracted_info.get("generation"))

        json_string = re.search(r'\{[^}]+\}', extracted_info.get("generation"), re.DOTALL).group()
        data = json.loads(json_string)
        
        print(data)
        
        # Convert the data to xlsx
        df = pd.DataFrame(data)
        df.to_excel("output.xlsx", index=False)
        # Upload the xlsx file to S3 named after the input file
        s3_client.upload_file("output.xlsx", bucket_name, f"{file_key}.xlsx")
        
        return JSONResponse(content=data)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/extract_Capital_Call")
async def extract_dataCapCall( file_key: str):
    try:
        # Fetch the file from S3
        file_key2 = file_key + ".pdf"

        s3_response = s3_client.get_object(Bucket=bucket_name, Key=file_key2)
        file_content = s3_response['Body'].read()

        # Process the file content
        content = extract_content_from_file(file_content)
        print(content)
        extracted_info = extract_structured_data_llama(content, data_points_capital_call)
        print(extracted_info.get("generation"))

        json_string = re.search(r'\{[^}]+\}', extracted_info.get("generation"), re.DOTALL).group()
        data = json.loads(json_string)
        
        print(data)
        # Convert the data to xlsx
        df = pd.DataFrame(data)
        df.to_excel("output.xlsx", index=False)
        # Upload the xlsx file to S3 named after the input file
        s3_client.upload_file("output.xlsx", bucket_name, f"{file_key}.xlsx")
        
        return JSONResponse(content=data)
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

import pandas as pd
import json
import boto3
import openpyxl
from openpyxl import load_workbook

# Function to convert JSON data to a spreadsheet
def json_to_spreadsheet(json_data, sheet_name, output_file):
    # Load JSON data
    data = json.loads(json_data)

    # Function to convert string to float, ignoring dollar symbols and converting to absolute value
    def to_float(value):
        if value == "none":
            return 0.0
        value = value.replace('$', '').replace(',', '')
        if '(' in value and ')' in value:
            value = value.replace('(', '-').replace(')', '')
        return abs(float(value))

    # Extract information
    customer_id = "Testwpf"
    fund_name = data["Information"]["FundName"]
    trade_date = data["Information"]["TradeDate"]
    settlement_date = data["Information"]["SettlementDate"]
    currency = data["Information"]["Currency"]

    # Get the value for DistributionProceeds
    private_equity_fund_sum = to_float(data["PrivateEquityFund"]["DistributionProceeds"])

    # Sum fields in IncomeAccounts
    income_accounts_sum = sum(to_float(data["IncomeAccounts"][k]) for k in data["IncomeAccounts"] if data["IncomeAccounts"][k] != "none")

    # Sum fields in ExpenseAccounts
    expense_accounts_sum = sum(to_float(data["ExpenseAccounts"][k]) for k in data["ExpenseAccounts"] if data["ExpenseAccounts"][k] != "none")

    # Generate transaction ID
    transaction_id = f"FS_AI_{fund_name}_{customer_id}_{trade_date}"
    # Prepare data for DataFrame
    rows = [
        [customer_id, "Dewey & Vera Goode", "", "A.I. General Journal", "DEBIT", "PRIVATE EQUITY", "", trade_date, settlement_date, "", 0.00, currency, fund_name, "Cash distribution", "", "", "", "", "", "", "", transaction_id],
        [customer_id, "Dewey & Vera Goode", "", "[Placeholder for Cash Account]", "DEBIT", "PRIVATE EQUITY", "", trade_date, settlement_date, "", private_equity_fund_sum, currency, fund_name, "Cash distribution", "", "", "", "", "", "", "", transaction_id],
        [customer_id, "Dewey & Vera Goode", "", f"{fund_name}/Distributions", "CREDIT", "PRIVATE EQUITY", "", trade_date, settlement_date, "", -private_equity_fund_sum, currency, fund_name, "Cash distribution", "", "", "", "", "", "", "", transaction_id],
        
        # Empty row for separation
        [""],
        
        [customer_id, "Dewey & Vera Goode", "", "A.I. General Journal", "DEBIT", "PRIVATE EQUITY", "", trade_date, settlement_date, "", 0.00, currency, fund_name, "Income & expense recognition", "", "", "", "", "", "", "", transaction_id],
        [customer_id, "Dewey & Vera Goode", "", f"{fund_name}/Distributions", "DEBIT", "PRIVATE EQUITY", "", trade_date, settlement_date, "", expense_accounts_sum - income_accounts_sum, currency, fund_name, "Income & expense recognition", "", "", "", "", "", "", "", transaction_id]
    ]

    # Add detailed entries from IncomeAccounts
    income_entries = {
        "Dividends": "Div",
        "STCapitalGainsLosses": "STCG",
        "LTCapitalGainsLosses": "LTCG",
        "Interest": "Interest Income"
    }

    for key, desc in income_entries.items():
        if data["IncomeAccounts"][key] != "none":
            rows.append([
                customer_id, "Dewey & Vera Goode", "", f"{fund_name} - {desc}", "CREDIT", "PRIVATE EQUITY", "", trade_date, settlement_date, "", -to_float(data["IncomeAccounts"][key]), currency, fund_name, "Income & expense recognition", "", "", "", "", "", "", "", transaction_id
            ])

    # Add detailed entries from ExpenseAccounts
    expense_entries = {
        "ManagementFees": "Management Fees",
        "InvestmentFees": "Investment Fees",
        "CarriedInterest": "Carried Interest",
        "LessBlockerExpenses": "Less Blocker Expenses",
        "DistributionWithheld": "Distribution Withheld",
        "OtherExpenses": "Other Expenses"
    }

    for key, desc in expense_entries.items():
        if data["ExpenseAccounts"][key] != "none":
            rows.append([
                customer_id, "Dewey & Vera Goode", "", f"{fund_name} - {desc}", "DEBIT", "PRIVATE EQUITY", "", trade_date, settlement_date, "", to_float(data["ExpenseAccounts"][key]), currency, fund_name, "Income & expense recognition", "", "", "", "", "", "", "", transaction_id
            ])

    # Create DataFrame
    columns = ["Customer Id", "Entity", "Custodian", "Account Number/Name", "Trans Type", "Security Type", "Symbol", "Trade Date", "Settlement Date", "Units", "Amount", "Currency Code", "Name", "Description", "Check Number", "Tran Sub-Type", "Accrued Interest", "STGL", "LTGL", "Original Sec Type", "Original Tran Type", "Transaction ID"]
    df = pd.DataFrame(rows, columns=columns)

    # Check if the output file exists
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        

    print(f"Data has been written to {output_file} in sheet {sheet_name}")
    
    # Upload the file to S3
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(output_file, bucket_name, f"{output_file}")
        print(f"File {output_file} uploaded to S3 bucket {bucket_name} ")
    except Exception as e:
        print(f"Error uploading file to S3: {str(e)}")
        
    return bucket_name, f"{output_file}"

@app.post("/extract_PE2")
# Function to process a single private equity document
def process_single_private_equity_document(bucket_name, file_key):
    
    output_file = file_key + '.xlsx'
    file_key = file_key + '.pdf'
    
    s3_client = boto3.client('s3')
    all_extracted_info = []

    try:
        s3_response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = s3_response['Body'].read()

        # Process the file content
        content = extract_content_from_file(file_content)

        template1 = f'''
        Please process the provided private equity document to extract essential transaction data, focusing on 'Debit' or 'Credit' financial movements. 
        The document has two types of transactions, one for the entire fund and the second one for the individual investor. Please extract the information that is related to the individual investor only:
        
        Extract the following details: Different variables should be extracted separately, including:
           Asset accounts: 
           - Commitment: The total amount of the capital commitment  (say "none" if not available)
           - Unfunded: The remaining commitment for additional investments in the future(say "none" if not available)
           - Net Distribution Proceeds: The net distributions made by the investment to the individual investor (to be received by the fund) (say "none" if not available)
           - Recallable: The amount of funds a private equity firm can request to be reinvested in the fund (say "none" if not available)
           - K-1 Income (Loss): K-1 income (say "none" if not available)
           - FMV Adjustment: The change in the Fair Market Value of the investment (say "none" if not available)
        Don't introduce any other information that is not relevant to the task.
        The Private Equity Document:
        {content}
        '''

        template2 = f'''
        Please process the provided private equity document to extract essential transaction data, focusing on 'Debit' or 'Credit' financial movements. We are interested in getting only the information on the partner its being emailed to. Follow these guidelines:
        The document has two types of transactions, one for the entire fund and the second one for the individual investor. Please extract the information that is related to the individual investor only:

        Extract the following details for only the client receiving the documents and not the general partners. Different variables should be extracted separately, including:
           Income accounts: 
           - Dividends: Income received from shares in companies (say "none" if not available)
           - Long Term Capital Gains (Losses): Profit or loss from selling investments held for more than a year (say "none" if not available)
           - Interest: Earnings from interest-bearing accounts. Only if it specifies that its an interest amount. (say "none" if not available)
           - Short Term Capital Gains (Losses): Profit or loss from selling investments held for less than a year (say "none" if not available)
           - Other Income: Any additional income not specified above. Do not include distribution (say "none" if not available)
           
           Expense accounts: 
           - Management Fees: Fees paid for investment management services (say "none" if not available)
           - Investment Fees: Charges for investment-related services, such as brokerage fees (say "none" if not available)
           - Carried Interest: Share of profits earned by investment managers (say "none" if not available)
           - Less Blocker Expenses: Deductions for expenses related to blocker entities in investments (say "none" if not available)
           - Distribution Withheld: The amount of distribution withheld by the fund (say "none" if not available)
           - Other Expenses: Any additional expenses not categorized above. Do not include net distribution proceeds (say "none" if not available)
        Don't introduce any other information that is not relevant to the task. Do not assume anything or make up any information.
        The Private Equity Document:
        {content}
        '''
        
        template3 = f'''
        Please process the provided private equity document to extract the name of private equity fund, the trade date, and the settlement date. Follow these guidelines:

        Extract the following details for only the client receiving the documents and not the general partners. Different variables should be extracted separately, including:
        
        - Private Equity Fund: The name of the private equity fund
        - Trade Date: The date of the trade in the format YYYYMMDD
        - Settlement Date: The date of the settlement in the format YYYYMMDD. Same as the trade date if it's a same-day settlement.
        - Currency: The currency code in which the transactions are denominated (e.g., USD, EUR, etc.). If not available, its USD by default.
        
        Don't introduce any other information that is not relevant to the task. Do not assume anything or make up any information.
        The Private Equity Document:
        {content}
        '''

        extracted_info1 = extract_structured_data_claude(template1)
        extracted_info2 = extract_structured_data_claude(template2)
        extracted_info3 = extract_structured_data_claude(template3)

        template4 = f'''
        Take the provided data and format it into a JSON object in the structure as shown below:
        {extracted_info1}
        {extracted_info2}
        {extracted_info3}
        
        Take the above information and format it into a JSON object in the structure as shown below:
        {{
          "Information": {{
            "FundName": "{{Private Equity Fund}}",
            "TradeDate": "{{Trade Date}}",
            "SettlementDate": "{{Settlement Date}}",
            "Currency":"{{the currency code}}"
          }},
          "PrivateEquityFund": {{
            "Commitment": "{{commitment}}",
            "Unfunded": "{{unfunded}}",
            "DistributionProceeds": "{{distribution_proceeds}}",
            "Recallable": "{{recallable}}",
            "K1IncomeLoss": "{{k1_income_loss}}",
            "FMVAdjustment": "{{fmv_adjustment}}"
          }},
          "IncomeAccounts": {{
            "Dividends": "{{dividends}}",
            "STCapitalGainsLosses": "{{st_capital_gains_losses}}",
            "LTCapitalGainsLosses": "{{lt_capital_gains_losses}}",
            "Interest": "{{interest}}",
            "OtherIncome": "{{other_income}}"
          }},
          "ExpenseAccounts": {{
            "ManagementFees": "{{management_fees}}",
            "InvestmentFees": "{{investment_fees}}",
            "CarriedInterest": "{{carried_interest}}",
            "LessBlockerExpenses": "{{less_blocker_expenses}}",
            "DistributionWithheld": "{{distribution_withheld}}",
            "OtherExpenses": "{{other_expenses}}"
          }}
        }}
        '''

        extracted_info4 = extract_structured_data_claude(template4)
        all_extracted_info.append(extracted_info4)

        # Convert extracted_info4 to JSON and call the json_to_spreadsheet function
        bucket_name, filename = json_to_spreadsheet(extracted_info4, file_key, output_file)

    except Exception as e:
        print(f"Error processing file {file_key}: {str(e)}")

    # Save the extracted information to a .txt file
    with open(output_file.replace('.xlsx', '.txt'), 'w') as f:
        for info in all_extracted_info:
            f.write(info + '\n')
            
    # Jsonify the output
    output_json = {
        "bucket_name": bucket_name,
        "filename": filename
    }
            
    return output_json
            
    

# # Define the bucket name and output file name
# bucket_name = 'forestai'
# output_file = 'extracted_info3.xlsx'

# # Manually input the file key each time
# file_key2 = 'Distribution_001'
# file_key = file_key2 + '.pdf'

# # Call the function
# process_single_private_equity_document(bucket_name, file_key, file_key2 + '.xlsx')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

