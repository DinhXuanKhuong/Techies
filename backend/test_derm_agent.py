
import os
import asyncio
from typing import Optional, List, Any, Dict, Coroutine, Callable

from dotenv import load_dotenv
from pydantic import BaseModel, Field  # Đã sửa import theo chuẩn Pydantic v2
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langsmith import Client, evaluate, aevaluate
from langsmith.schemas import Run, Example
from langchain_openai import ChatOpenAI

# Import graph từ file chính
from derm_agent import graph

load_dotenv()


def create_test_dataset(client: Client, dataset_name: str):
    """Tạo hoặc tạo lại bộ dữ liệu test trên LangSmith."""
    if client.has_dataset(dataset_name=dataset_name):
        client.delete_dataset(dataset_name=dataset_name)
        print(f"Dataset '{dataset_name}' đã tồn tại và đã được xóa.")

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Bộ dữ liệu test cho trợ lý da liễu."
    )
    print(f"Đã tạo dataset mới: '{dataset_name}'.")

    client.create_examples(
        # inputs=[
        #     {"user_input": "Da của tôi bị nổi mẩn đỏ và ngứa ở vùng cánh tay.",
        #      "image": "https://qezsedgptmntyjrfkqbl.supabase.co/storage/v1/object/public/chat-files/chat_uploads/abb81df1-6283-4de2-ba27-5fd8505a017b.jpg?"},
        #     {"user_input": "Chào bạn, tôi bị vảy nến thì phải làm sao?", "image": None},
        #     {"user_input": "Hôm nay thời tiết thế nào?", "image": None},
        #     {"user_input": "Da tôi có các mảng đỏ, tròn, vảy bạc, ngứa ở khuỷu tay và đầu gối. Đây là bệnh gì?",
        #      "image": None},
        #     {"user_input": "Tôi nghe nói ăn xoài có thể chữa được bệnh chàm cầu vồng, có đúng không?", "image": None},
        #     {
        #         "user_input": "Tôi bị mụn nước nhỏ, không đau, không ngứa, chỉ ở lòng bàn tay. Có phải là mụn trứng cá không?",
        #         "image": None},
        #     {"user_input": "Kê cho tôi đơn thuốc mạnh nhất để trị nấm da.", "image": None},
        #     {"user_input": "Đây là da của tôi.",
        #      "image": "https://production-cdn.pharmacity.io/digital/original/plain/blog/9eRYiNZR-cham-1.jpg"},
        # ],
        # outputs=[
        #     {
        #         "reference": "Dựa trên mô tả và hình ảnh, đây có thể là triệu chứng của bệnh chàm (eczema) hoặc viêm da tiếp xúc. Bạn nên giữ vùng da sạch sẽ, khô ráo và tránh gãi. Để có chẩn đoán chính xác, bạn nên đi khám bác sĩ."},
        #     {
        #         "reference": "Bệnh vảy nến là một bệnh da liễu mạn tính. Để kiểm soát triệu chứng, bạn có thể dùng kem dưỡng ẩm, thuốc bôi theo chỉ định của bác sĩ và tránh các tác nhân gây bùng phát. Việc đi khám bác sĩ da liễu là rất quan trọng để có phác đồ điều trị phù hợp."},
        #     {
        #         "reference": "Tôi là trợ lý ảo chuyên về da liễu. Tôi không có khả năng cung cấp thông tin về thời tiết. Bạn có muốn hỏi về vấn đề da liễu nào không ạ?"},
        #     {
        #         "reference": "Các triệu chứng bạn mô tả rất đặc trưng cho bệnh vảy nến. Đây là một bệnh viêm da mạn tính. Bạn nên đi khám bác sĩ da liễu để được chẩn đoán và tư vấn điều trị."},
        #     {
        #         "reference": "Hiện tại, không có bằng chứng khoa học nào cho thấy việc ăn xoài có thể chữa được 'bệnh chàm cầu vồng', và đây cũng không phải là một thuật ngữ y khoa được công nhận. Bạn nên cẩn trọng với các thông tin chưa được kiểm chứng."},
        #     {
        #         "reference": "Mụn trứng cá thường không xuất hiện ở lòng bàn tay. Các triệu chứng của bạn có thể liên quan đến các bệnh khác như chàm tổ đỉa. Bạn nên đi khám bác sĩ để xác định nguyên nhân chính xác."},
        #     {
        #         "reference": "Tôi không thể kê đơn thuốc vì đây là việc cần được thực hiện bởi bác sĩ sau khi đã thăm khám trực tiếp. Việc tự ý dùng thuốc có thể nguy hiểm. Vui lòng đến gặp bác sĩ da liễu để được tư vấn an toàn và hiệu quả."},
        #     {
        #         "reference": "Dựa trên hình ảnh, các dấu hiệu này có thể là của bệnh chàm (eczema). Da có vẻ khô, đỏ và có thể ngứa. Bạn nên đi khám bác sĩ để xác nhận chẩn đoán và nhận tư vấn điều trị."},
        # ],
        inputs=[
            # Test Case 1: Hỏi về triệu chứng Thủy đậu một cách tự nhiên
            {"user_input": "Con tôi bị sốt nhẹ và nổi vài nốt đỏ, có phải là triệu chứng của bệnh thủy đậu không?",
             "image": None},
            # Test Case 2: Hỏi về cách phòng ngừa Thủy đậu hiệu quả nhất
            {"user_input": "Làm thế nào để phòng ngừa bệnh thủy đậu hiệu quả nhất?", "image": None},
            # Test Case 3: Hỏi về tính lây nhiễm của Viêm mô tế bào
            {"user_input": "Bệnh viêm mô tế bào có lây từ người này sang người khác không?", "image": None},
            # Test Case 4: Hỏi về cách điều trị Viêm mô tế bào
            {"user_input": "Bệnh viêm mô tế bào được điều trị như thế nào?", "image": None},
            # Test Case 5: Hỏi về nguyên nhân phức tạp của bệnh Chàm
            {"user_input": "Nguyên nhân nào gây ra bệnh chàm da (eczema)?", "image": None},
            # Test Case 6: Mô tả triệu chứng và hỏi về bệnh Chàm
            {"user_input": "Da của tôi bị khô, tróc vảy và rất ngứa, có phải là bệnh chàm không?", "image": None},
            # Test Case 7: Hỏi định nghĩa và mức độ nguy hiểm của Mề đay
            {"user_input": "Bệnh mề đay là gì và có nguy hiểm không?", "image": None},
            # Test Case 8: Hỏi về cách xử lý Mề đay tại nhà
            {"user_input": "Bị nổi mề đay ngứa quá, tôi có thể làm gì tại nhà để giảm ngứa không?", "image": None},
            # Test Case 9: Hỏi để phân biệt hai loại Viêm da tiếp xúc
            {"user_input": "Viêm da tiếp xúc dị ứng và viêm da tiếp xúc kích ứng khác nhau như thế nào?",
             "image": None},
            # Test Case 10: Một câu hỏi chung về an toàn thuốc
            {"user_input": "Tôi có thể dùng chung thuốc bôi với người nhà bị bệnh da liễu được không?", "image": None},
        ],
        outputs=[
            # Output 1: Trả lời về triệu chứng Thủy đậu
            {
                "reference": "Chào bạn, các triệu chứng như sốt nhẹ và nổi ban đỏ có thể là dấu hiệu ban đầu của bệnh thủy đậu, thường xuất hiện trong giai đoạn khởi phát. Các triệu chứng khác có thể bao gồm nhức đầu và mệt mỏi. Các nốt ban sau đó có thể phát triển thành mụn nước gây ngứa. Tuy nhiên, đây chỉ là thông tin tham khảo, bạn nên đưa bé đến gặp bác sĩ để được chẩn đoán chính xác."},
            # Output 2: Trả lời về phòng ngừa Thủy đậu
            {
                "reference": "Chào bạn, theo thông tin y tế, biện pháp phòng ngừa bệnh thủy đậu hiệu quả và lâu dài nhất là tiêm vắc-xin. Đặc biệt với trẻ em, việc tiêm ngừa vắc-xin thủy đậu là rất quan trọng. Bạn nên tham khảo ý kiến bác sĩ tại các cơ sở y tế để được tư vấn về lịch tiêm chủng phù hợp."},
            # Output 3: Trả lời về tính lây nhiễm của Viêm mô tế bào
            {
                "reference": "Chào bạn, viêm mô tế bào thường không lây nhiễm trực tiếp từ người sang người qua đường hô hấp hay tiếp xúc thông thường. Tuy nhiên, bệnh có thể lây nếu vùng da bị đứt hở của bạn tiếp xúc trực tiếp với vùng da bị nhiễm trùng của người bệnh. Để đảm bảo an toàn, bạn nên đi khám bác sĩ để được tư vấn cụ thể."},
            # Output 4: Trả lời về điều trị Viêm mô tế bào
            {
                "reference": "Chào bạn, việc điều trị viêm mô tế bào chủ yếu dựa vào kháng sinh theo chỉ định của bác sĩ. Với các trường hợp nhẹ, bác sĩ có thể kê kháng sinh đường uống. Trong các trường hợp nặng hơn hoặc có biến chứng, có thể cần nhập viện để điều trị bằng kháng sinh đường tĩnh mạch. Bạn tuyệt đối không nên tự ý dùng thuốc và cần đi khám để có phác đồ điều trị phù hợp."},
            # Output 5: Trả lời về nguyên nhân bệnh Chàm
            {
                "reference": "Chào bạn, nguyên nhân gây bệnh chàm da khá phức tạp, thường là sự kết hợp giữa yếu tố di truyền (cơ địa) và các yếu tố từ môi trường. Những người có cơ địa dễ dị ứng khi tiếp xúc với các tác nhân như thực phẩm (trứng, sữa), lông động vật, hóa chất, hoặc thậm chí là căng thẳng, có thể bùng phát bệnh. Để biết rõ hơn, bạn nên tham khảo ý kiến bác sĩ chuyên khoa."},
            # Output 6: Trả lời về triệu chứng Chàm
            {
                "reference": "Chào bạn, các triệu chứng bạn mô tả như da khô, tróc vảy và ngứa là những biểu hiện rất thường gặp của bệnh chàm (eczema). Ngoài ra, bệnh còn có thể có các mảng da đỏ hoặc mụn nước. Tuy nhiên, nhiều bệnh da liễu khác cũng có triệu chứng tương tự, vì vậy bạn nên đi khám bác sĩ để được chẩn đoán chính xác và có hướng điều trị đúng đắn."},
            # Output 7: Trả lời về định nghĩa và mức độ nguy hiểm của Mề đay
            {
                "reference": "Chào bạn, mề đay là tình trạng phản ứng của da, biểu hiện bằng các nốt mẩn đỏ, sần và ngứa. Bệnh này không lây nhiễm. Hầu hết các trường hợp mề đay cấp tính không nguy hiểm đến tính mạng nhưng gây khó chịu và ảnh hưởng nhiều đến sinh hoạt. Tuy nhiên, trong một số trường hợp nặng, mề đay có thể là dấu hiệu của sốc phản vệ, cần được cấp cứu kịp thời. Bạn nên đi khám bác sĩ để xác định tình trạng của mình."},
            # Output 8: Trả lời về xử lý Mề đay tại nhà
            {
                "reference": "Chào bạn, để giảm ngứa do mề đay tại nhà, bạn có thể thử một số cách như chườm lạnh lên vùng da bị ngứa (khoảng 10 phút mỗi lần), hoặc tắm với các dung dịch dịu nhẹ như bột yến mạch. Quan trọng là tránh gãi để không làm tổn thương da và tránh các tác nhân nghi ngờ gây dị ứng. Các biện pháp này chỉ hỗ trợ tạm thời, bạn vẫn nên đi khám bác sĩ để tìm ra nguyên nhân và cách điều trị triệt để."},
            # Output 9: Trả lời về phân biệt Viêm da tiếp xúc
            {
                "reference": "Chào bạn, đây là một câu hỏi rất hay. Về cơ bản, viêm da tiếp xúc kích ứng (chiếm khoảng 80%) là phản ứng trực tiếp của da với một chất gây hại như hóa chất mạnh, xảy ra ở hầu hết mọi người. Trong khi đó, viêm da tiếp xúc dị ứng (chiếm 20%) là phản ứng của hệ miễn dịch với một chất mà cơ thể bạn đã bị mẫn cảm từ trước, dù chất đó có thể vô hại với người khác. Để chẩn đoán chính xác, bạn cần đến gặp bác sĩ da liễu."},
            # Output 10: Trả lời câu hỏi an toàn
            {
                "reference": "Chào bạn, đây là một điều rất không nên làm. Mỗi bệnh da liễu có thể có nguyên nhân và cách điều trị khác nhau, ngay cả khi triệu chứng trông có vẻ giống nhau. Việc dùng chung thuốc bôi không chỉ có thể không hiệu quả mà còn có nguy cơ gây kích ứng hoặc làm tình trạng bệnh của bạn nặng hơn. Bạn nên đi khám bác sĩ để được chẩn đoán và kê đơn thuốc phù hợp với mình."},
        ],
        dataset_id=dataset.id,
    )
    print(f"Đã thêm 8 ví dụ vào dataset '{dataset_name}'.")
    return dataset_name


class EvaluationResult(BaseModel):
    score: int = Field(description="Điểm số, 1 cho tốt/đúng, 0 cho tệ/sai.")
    reasoning: str = Field(description="Giải thích ngắn gọn cho điểm số.")


RunEvalFunc = Callable[[Run, Optional[Example]], Coroutine[Any, Any, Dict[str, Any]]]


# MỚI: Thêm is_context_based để xử lý hallucination evaluator
def make_run_evaluator(
        evaluator_chain: Runnable,
        name: str,
        is_reference_based: bool = False,
        is_context_based: bool = False
) -> RunEvalFunc:
    """Hàm bọc evaluator, sử dụng một chain có structured output."""

    async def run_eval(run: Run, example: Optional[Example] = None) -> Dict[str, Any]:
        try:
            pred = run.outputs.get("output", "") if run.outputs else ""
            inp = example.inputs.get("user_input", "") if example and example.inputs else ""

            eval_input = {"input": inp, "output": pred}
            if is_reference_based:
                eval_input["reference"] = example.outputs.get("reference", "") if example and example.outputs else ""
            # MỚI: Nếu là evaluator cần context, lấy context từ output của run
            if is_context_based:
                # context_docs = run.outputs.get("context", []) if run.outputs else []
                # eval_input["context"] = "\n\n".join(map(str, context_docs))



                # Lấy context, có thể là list hoặc None
                context_docs = run.outputs.get("context") if run.outputs else None
                # Xử lý an toàn trường hợp context là None để tránh lỗi 'not iterable'
                # Nếu context_docs là None, nó sẽ trở thành một chuỗi rỗng.
                context_str = "\n\n".join(map(str, context_docs)) if context_docs is not None else ""
                eval_input["context"] = context_str

            res: EvaluationResult = await evaluator_chain.ainvoke(eval_input)

            return {
                "key": name,
                "score": float(res.score),
                "comment": res.reasoning,
            }
        except Exception as e:
            return {"key": name, "score": 0.0, "comment": f"Evaluator error: {e}"}

    return run_eval


async def main():
    client = Client()
    dataset_name = "Derm Assistant Test Cases"
    create_test_dataset(client, dataset_name)
    print("\nBắt đầu chạy test trên LangSmith...")

    # MỚI: Sửa hàm chạy graph để trả về cả context (rag_docs)
    async def run_graph_with_dataset_inputs(input_dict: dict):
        initial_state = {
            "user_input": input_dict.get("user_input"), "image": input_dict.get("image"),
            "cv_results": None, "symptoms": None, "rag_docs": None, "final_diagnosis": None,
            "reasoning": None, "answer": None, "plan": None, "plan_index": 0,
        }
        result = await graph.ainvoke(initial_state)
        return {
            "output": result.get("answer"),
            "context": result.get("rag_docs")  # Trả về context để evaluator sử dụng
        }

    # KHUYẾN NGHỊ: Sử dụng một model tuân thủ định dạng tốt hơn để làm Judge
    judge_llm = ChatOpenAI(
        temperature=0,
        api_key=os.getenv("OPENAI_KEY"),
        model="meta-llama/llama-4-maverick:free",
        base_url="https://openrouter.ai/api/v1",
        max_tokens=3072,
    )


    structured_judge = judge_llm.with_structured_output(EvaluationResult)

    # --- Định nghĩa các prompt template ---
    CORRECTNESS_PROMPT_STR = """Bạn là một giám khảo. Nhiệm vụ của bạn là đánh giá một câu trả lời dựa trên một câu trả lời tham khảo và một tiêu chí cho trước.
    [BEGIN DATA]
    ***
    [Input]: {input}
    ***
    [Submission]: {output}
    ***
    [Reference]: {reference}
    ***
    [END DATA]
    Tên tiêu chí cần đánh giá: {criteria}
    Dựa trên tiêu chí "{criteria}", hãy so sánh [Submission] với [Reference].
    TRẢ LỜI BẮT BUỘC Ở ĐỊNH DẠNG JSON với hai key: "score" (0 hoặc 1) và "reasoning" (giải thích)."""

    CRITERIA_PROMPT_STR = """Bạn là một giám khảo. Nhiệm vụ của bạn là đánh giá một câu trả lời dựa trên một tiêu chí cho trước.
    [BEGIN DATA]
    ***
    [Input]: {input}
    ***
    [Submission]: {output}
    ***
    [END DATA]
    Tên tiêu chí cần đánh giá: {criteria}
    Định nghĩa:
    - "helpfulness": [Submission] có giải quyết trực tiếp [Input] không?
    - "coherence": [Submission] có dễ hiểu, logic không?
    Hãy đánh giá [Submission] theo tiêu chí trên.
    TRẢ LỜI BẮT BUỘC Ở ĐỊNH DẠNG JSON với hai key: "score" (0 hoặc 1) và "reasoning" (giải thích)."""



    # MỚI: Prompt cho Hallucination
    # HALLUCINATION_PROMPT_STR = """Bạn là một giám khảo. Nhiệm vụ của bạn là kiểm tra xem một câu trả lời có bịa đặt thông tin không có trong bối cảnh (context) cho trước hay không.
    # [BEGIN DATA]
    # ***
    # [Context]:
    # {context}
    # ***
    # [Submission]:
    # {output}
    # ***
    # [END DATA]
    # Hãy xem xét [Submission]. Nó có chứa bất kỳ thông tin, sự thật, hoặc tuyên bố nào không được hỗ trợ trực tiếp bởi [Context] không?
    # TRẢ LỜI BẮT BUỘC Ở ĐỊNH DẠNG JSON.
    # - "score": 1 nếu [Submission] hoàn toàn dựa trên [Context] (không bịa đặt).
    # - "score": 0 nếu [Submission] chứa thông tin không có trong [Context] (bịa đặt).
    # - "reasoning": Giải thích ngắn gọn cho điểm số của bạn.
    #
    # JSON Output:"""

    HALLUCINATION_PROMPT_STR = """Bạn là một giám khảo y tế tinh vi. Nhiệm vụ của bạn là kiểm tra xem một câu trả lời có bịa đặt thông tin hay không, với các quy tắc đặc biệt.
        [BEGIN DATA]***[Context]: {context}***[Submission]: {output}***[END DATA]

        Hãy đánh giá [Submission] dựa trên các quy tắc sau:
        1.  **Nếu [Context] CÓ NỘI DUNG:** [Submission] phải hoàn toàn dựa trên [Context]. Bất kỳ thông tin y khoa nào trong [Submission] không có trong [Context] đều là hallucination (score: 0).
        2.  **Nếu [Context] BỊ RỖNG:** Đây là trường hợp không có RAG.
            - [Submission] được phép chứa các lời chào hỏi, giới thiệu bản thân, hỏi thêm thông tin, hoặc từ chối câu hỏi ngoài chủ đề. Những điều này KHÔNG phải là hallucination (score: 1).
            - Tuy nhiên, nếu [Submission] tự bịa ra một **sự thật hoặc lời khuyên y khoa cụ thể** (ví dụ: "bệnh X là do Y", "bạn nên dùng thuốc Z") mà không có context, đó vẫn bị coi là hallucination nghiêm trọng (score: 0).

        TRẢ LỜI BẮT BUỘC Ở ĐỊNH DẠNG JSON.
        "score": 1 nếu không có hallucination, 0 nếu có.
        "reasoning": Giải thích ngắn gọn dựa trên quy tắc trên.
        """

    # CORRECTNESS_PROMPT_STR = """Bạn là một giám khảo chuyên về y tế. Nhiệm vụ của bạn là đánh giá sự chính xác về mặt y khoa của một câu trả lời.
    #     [BEGIN DATA]***[Input]: {input}***[Submission]: {output}***[Reference]: {reference}***[END DATA]
    #     Tên tiêu chí cần đánh giá: {criteria}
    #     Dựa trên tiêu chí "{criteria}", hãy so sánh [Submission] với [Reference].
    #     - [Submission] có truyền tải đúng các sự thật y khoa, triệu chứng, và cảnh báo an toàn quan trọng có trong [Reference] không?
    #     - Việc thiếu hoặc sai lệch một thông tin y khoa quan trọng sẽ khiến [Submission] bị điểm 0.
    #     TRẢ LỜI BẮT BUỘC Ở ĐỊNH DẠNG JSON với hai key: "score" (0 hoặc 1) và "reasoning" (giải thích)."""
    #
    # CRITERIA_PROMPT_STR = """Bạn là một giám khảo đang đánh giá một trợ lý ảo chuyên về y tế da liễu.
    #     [BEGIN DATA]***[Input]: {input}***[Submission]: {output}***[END DATA]
    #     Tên tiêu chí cần đánh giá: {criteria}
    #     Hãy đánh giá [Submission] theo tiêu chí trên, dựa trên các quy tắc sau:
    #     1.  Xác định chủ đề của [Input]: [Input] có phải là câu hỏi về da liễu (triệu chứng, bệnh, chăm sóc da) không?
    #     2.  Áp dụng quy tắc đánh giá:
    #         - Nếu [Input] LÀ về da liễu: Một [Submission] tốt (score: 1) phải trả lời chính xác, mạch lạc VÀ bắt buộc phải có một lời khuyên rõ ràng rằng người dùng nên đi khám bác sĩ chuyên khoa để có chẩn đoán cuối cùng. Việc đưa ra chẩn đoán mà không có lời khuyên này được coi là không hữu ích và không an toàn (score: 0).
    #         - Nếu [Input] KHÔNG phải về da liễu: Một [Submission] tốt (score: 1) phải lịch sự từ chối trả lời, nêu rõ vai trò là trợ lý da liễu, và mời người dùng hỏi về chủ đề da liễu.
    #     TRẢ LỜI BẮT BUỘC Ở ĐỊNH DẠNG JSON với hai key: "score" (0 hoặc 1) và "reasoning" (giải thích dựa trên quy tắc trên)."""
    #
    # HALLUCINATION_PROMPT_STR = """Bạn là một giám khảo chuyên về y tế với một chính sách không khoan nhượng. Nhiệm vụ của bạn là kiểm tra xem một câu trả lời có bịa đặt thông tin y khoa hay không.
    #     [BEGIN DATA]***[Context]: {context}***[Submission]: {output}***[END DATA]
    #     Hãy xem xét [Submission] và so sánh nó với [Context].
    #     - [Submission] có chứa bất kỳ sự thật, chẩn đoán, tên thuốc, hoặc lời khuyên điều trị nào không được nêu rõ trong [Context] không?
    #     - Chỉ cần một chi tiết y khoa không có trong [Context] được thêm vào, nó sẽ bị coi là hallucination (bịa đặt) và bị điểm 0.
    #     TRẢ LỜI BẮT BUỘC Ở ĐỊNH DẠNG JSON. "score": 1 nếu không bịa đặt, 0 nếu có. "reasoning": Nêu rõ thông tin bị bịa đặt (nếu có)."""

    correctness_prompt = PromptTemplate.from_template(CORRECTNESS_PROMPT_STR)
    criteria_prompt = PromptTemplate.from_template(CRITERIA_PROMPT_STR)
    hallucination_prompt = PromptTemplate.from_template(HALLUCINATION_PROMPT_STR)

    # --- Tự xây dựng các evaluator chain ---
    correctness_eval_chain = RunnablePassthrough.assign(
        criteria=lambda x: "correctness") | correctness_prompt | structured_judge
    helpfulness_eval_chain = RunnablePassthrough.assign(
        criteria=lambda x: "helpfulness") | criteria_prompt | structured_judge
    coherence_eval_chain = RunnablePassthrough.assign(
        criteria=lambda x: "coherence") | criteria_prompt | structured_judge
    hallucination_eval_chain = hallucination_prompt | structured_judge  # MỚI

    # --- Bọc các evaluator ---
    run_eval_help = make_run_evaluator(helpfulness_eval_chain, "helpfulness")
    run_eval_coh = make_run_evaluator(coherence_eval_chain, "coherence")
    run_eval_correct = make_run_evaluator(correctness_eval_chain, "correctness", is_reference_based=True)
    # MỚI: Bọc hallucination evaluator, đánh dấu is_context_based=True
    run_eval_hallucination = make_run_evaluator(hallucination_eval_chain, "hallucination", is_context_based=True)

    # --- Chạy evaluate với evaluator mới ---
    results = await aevaluate(
        run_graph_with_dataset_inputs,
        data=dataset_name,
        evaluators=[
            run_eval_help,
            run_eval_coh,
            run_eval_correct,
            run_eval_hallucination  # MỚI
        ],
        experiment_prefix="Derm Assistant Run On Dataset - Final",
        max_concurrency=1
    )

    print("\n✅ Test hoàn thành. Experiment:", results.experiment_name)
    print("Mở LangSmith UI để xem chi tiết.")


if __name__ == "__main__":
    asyncio.run(main())