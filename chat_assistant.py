from openai import OpenAI
import time

CHAT_ASSISTANT_NAME = "Warehouse Data Analyst"  # Name of assistant
GPT_MODEL = "gpt-4o"
ASSISTANT_INSTRUCTIONS = """You are a personal 3PL data analyst specialized in analyzing data from multiple warehouse operation logs. Your main task is to assist by analyzing data and providing insights based on user questions.

When a user asks a question that requires analysis, follow these steps:

1. **Clarify the Question**: If the question or request seems unclear, ask follow-up questions to gather more specific information. Remember to use a conversational approach to ensure users feel comfortable and understood. 
2. **Data Handling**: Write and execute code to answer the user's question using the data provided in the uploaded files. The data is spread across multiple files, so ensure you run your analysis across all datasets. Use efficient data processing techniques to handle these large files effectively.
3. **Header Definitions**: The dataset includes the following headers, each with a specific meaning:
   - **position_id**: The job's unique ID
   - **action**: Type of action completed
   - **quantity**: Number of items completed during the action
   - **sec_elapsed**: Time taken (in seconds) by the associate to complete the action
   - **full_name**: Name of the associate
   - **associate_oid**: Unique ID for the associate
   - **calendar_date**: Date of the action
   - **action_started_local_time**: Start time of the action
   - **action_completed_local_time**: Completion time of the action

4. **Accuracy and Efficiency**: Focus on accuracy in the results and apply efficient data processing strategies, especially considering the data's large size.

5. **Communication Style**: Maintain a conversational and approachable tone throughout your interactions with the user, to cater to various levels of technical and analytical understanding.

# Output Format

Provide your analysis result in a user-friendly summary, supported by relevant statistics, charts, or tables if necessary. Use markdown for formatting the response where applicable.

# Examples

**Example 1: Analytical Question**
- **User Input**: "How many actions were completed by the associate named Alex on June 15?"
- **Steps**:
  - Aggregate data from all files.
  - Filter by `full_name` = "Alex" and `calendar_date` = "June 15".
  - Summarize action counts.
- **Output**: "On June 15, Alex completed 47 actions."

**Example 2: Timing Analysis**
- **User Input**: "What was the average time taken per item for position ID 12345?"
- **Steps**:
  - Filter data for `position_id` = 12345.
  - Calculate the average time (`sec_elapsed`) divided by `quantity`.
- **Output**: "The average time taken per item for position ID 12345 is 3.5 seconds."

# Notes

- Ensure consistent data merging across multiple files for accurate results.
- Ask follow-up questions to clarify ambiguous requests.
- If handling extremely large datasets, consider streaming or chunk processing to optimize performance."""

# Delay between response characters stream (in seconds)
CHARACTER_STREAM_DELAY = 0.005           

class ChatAssistant:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.assistant = None
        self.file_ids = []
        self.thread = None

        self.get_files()
        self.get_assistant()

    def get_assistant(self):
        assistants = self.client.beta.assistants.list()
        for ass in assistants:
            if ass.name == CHAT_ASSISTANT_NAME:
                self.assistant = self.client.beta.assistants.retrieve(ass.id)
                print(f"Retrieved chat assistant: {self.assistant.id} - {self.assistant.name}")
                self.update_assistant()
                
        if self.assistant is None:
            self.assistant = self.client.beta.assistants.create (
                name = CHAT_ASSISTANT_NAME,
                instructions = ASSISTANT_INSTRUCTIONS,
                model = GPT_MODEL,
                tools = [{'type': 'code_interpreter'}],
                tool_resources={
                    "code_interpreter": {
                    "file_ids": self.file_ids
                    }
                }
            )
            print(f"Created chat assistant: {self.assistant.id} - {self.assistant.name}")

    def get_files(self):
        file_list = self.client.files.list()
        self.file_ids = [file_list.data[0].id]
        # self.file_ids = [file_list.data[0].id, file_list.data[1].id, file_list.data[2].id, file_list.data[3].id]
        print(f"Got files: {self.file_ids}")

    def update_assistant(self):
        self.assistant = self.client.beta.assistants.update(
            self.assistant.id,
            instructions = ASSISTANT_INSTRUCTIONS,
            model = GPT_MODEL,
            tools = [{'type': 'code_interpreter'}],
            tool_resources={
                "code_interpreter": {
                "file_ids": self.file_ids
                }
            }
        )
        print("Updated chat assistant.")

    def renew_thread(self):
        self.thread = self.client.beta.threads.create()
        print(f"Renewed chat assistant thread: {self.thread.id}")

    def get_response(self, prompt):
        print(f"New chat assistant prompt: {prompt}")

        if self.thread is None:
            self.renew_thread()

        message = self.client.beta.threads.messages.create(
            thread_id = self.thread.id,
            role = 'user',
            content=prompt
        )

        run = self.client.beta.threads.runs.create(
            thread_id = self.thread.id,
            assistant_id = self.assistant.id,
            stream=True
        )

        for chunk in run:
            if chunk.data.object == 'thread.message' and chunk.data.content:
                msg = chunk.data
                for content in msg.content:
                    # Check if the content is text
                    if content.type == 'text':
                        print(f"{msg.role}: {content.text.value}")

                        # Send response back (delay each character for typewriter effect)
                        for char in content.text.value:
                            yield char
                            time.sleep(CHARACTER_STREAM_DELAY)

                        # Check and print details about annotations if they exist
                        if content.text.annotations:
                            for annotation in content.text.annotations:
                                print(f"Annotation Text: {annotation.text}")
                                print(f"File_Id: {annotation.file_path.file_id}")
                                annotation_data = self.client.files.content(annotation.file_path.file_id)
                                annotation_data_bytes = annotation_data.read()

                                filename = annotation.text.split('/')[-1]

                                with open(f"{filename}", "wb") as file:
                                    file.write(annotation_data_bytes)

                    # Check if the content is an image file and print its file ID and name
                    elif content.type == 'image_file':
                        print(f"Image File ID: {content.image_file.file_id}")
                        yield f"Image File ID: {content.image_file.file_id}"

            if hasattr(chunk.data, 'status'):
                print(f"Status: {chunk.data.status}")