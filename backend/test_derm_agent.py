
import os
import asyncio
from typing import Optional, List, Any, Dict, Coroutine, Callable

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field  # Đã sửa import theo chuẩn Pydantic v2
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langsmith import Client, evaluate, aevaluate
from langsmith.schemas import Run, Example
from langchain_openai import ChatOpenAI

# Import graph từ file chính
from derm_agent import graph

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
        inputs=[
            # Các case cũ có ảnh
            {"user_input": "Da của tôi bị nổi mẩn đỏ và ngứa ở vùng cánh tay.",
             "image": "https://qezsedgptmntyjrfkqbl.supabase.co/storage/v1/object/public/chat-files/chat_uploads/abb81df1-6283-4de2-ba27-5fd8505a017b.jpg?"},
            {"user_input": "Đây là da của tôi.",
             "image": "https://production-cdn.pharmacity.io/digital/original/plain/blog/9eRYiNZR-cham-1.jpg"},

            # 10 test case đầu tiên
            {"user_input": "Con tôi bị sốt nhẹ và nổi vài nốt đỏ, có phải là triệu chứng của bệnh thủy đậu không?",
             "image": None},
            {"user_input": "Làm thế nào để phòng ngừa bệnh thủy đậu hiệu quả nhất?", "image": None},
            {"user_input": "Bệnh viêm mô tế bào có lây từ người này sang người khác không?", "image": None},
            {"user_input": "Bệnh viêm mô tế bào được điều trị như thế nào?", "image": None},
            {"user_input": "Nguyên nhân nào gây ra bệnh chàm da (eczema)?", "image": None},
            {"user_input": "Da của tôi bị khô, tróc vảy và rất ngứa, có phải là bệnh chàm không?", "image": None},
            {"user_input": "Bệnh mề đay là gì và có nguy hiểm không?", "image": None},
            {"user_input": "Bị nổi mề đay ngứa quá, tôi có thể làm gì tại nhà để giảm ngứa không?", "image": None},
            {"user_input": "Viêm da tiếp xúc dị ứng và viêm da tiếp xúc kích ứng khác nhau như thế nào?",
             "image": None},
            {"user_input": "Tôi có thể dùng chung thuốc bôi với người nhà bị bệnh da liễu được không?", "image": None},

            # 30 test case mới
            {"user_input": "Bệnh thủy đậu có để lại sẹo không và làm sao để hạn chế sẹo?", "image": None},
            {"user_input": "Người lớn bị thủy đậu thì có nặng hơn trẻ em không?", "image": None},
            {"user_input": "Khi bị thủy đậu thì cần phải cách ly trong bao lâu?", "image": None},
            {"user_input": "Viêm mô tế bào có thể xảy ra ở vùng mặt không?", "image": None},
            {"user_input": "Nếu không chữa viêm mô tế bào kịp thời thì sẽ bị gì?", "image": None},
            {"user_input": "Tại sao bệnh chàm của tôi lại nặng hơn vào mùa đông?", "image": None},
            {"user_input": "Bị chàm thì nên dùng loại sữa tắm hay xà phòng nào?", "image": None},
            {"user_input": "Trẻ sơ sinh bị chàm thì có tự hết không?", "image": None},
            {"user_input": "Ăn uống có ảnh hưởng đến bệnh chàm không, tôi nên kiêng gì?", "image": None},
            {"user_input": "Tôi bị nổi mề đay mỗi khi căng thẳng, có cách nào khắc phục không?", "image": None},
            {"user_input": "Một đợt nổi mề đay thường kéo dài trong bao lâu?", "image": None},
            {"user_input": "Làm sao để phân biệt giữa nổi mề đay và phát ban thông thường?", "image": None},
            {"user_input": "Phụ nữ mang thai bị nổi mề đay có ảnh hưởng đến em bé không?", "image": None},
            {"user_input": "Tôi đeo dây chuyền giả thì bị ngứa và nổi đỏ ở cổ, có phải viêm da tiếp xúc không?",
             "image": None},
            {"user_input": "Sau khi đi làm vườn, tay tôi bị ngứa và nổi mụn nước, có phải do dị ứng lá cây không?",
             "image": None},
            {"user_input": "Viêm da tiếp xúc thì bôi thuốc bao lâu sẽ khỏi?", "image": None},
            {"user_input": "Bệnh zona có phải do virus thủy đậu gây ra không?", "image": None},
            {"user_input": "Bệnh vảy nến có chữa dứt điểm được không?", "image": None},
            {"user_input": "Viêm da cơ địa và chàm có phải là một không?", "image": None},
            {"user_input": "Làm sao để phân biệt nấm da và bệnh chàm?", "image": None},
            {"user_input": "Tôi đã bôi thuốc trị mề đay mà bác sĩ kê nhưng vẫn không đỡ ngứa.", "image": None},
            {"user_input": "Dùng lá trầu không để tắm có chữa được bệnh chàm không?", "image": None},
            {"user_input": "Da tôi có vấn đề.", "image": None},
            {"user_input": "Xem giúp tôi cái ảnh này là bệnh gì.",
             "image": "https://production-cdn.pharmacity.io/digital/original/plain/blog/9eRYiNZR-cham-1.jpg"},
            # Sử dụng lại ảnh chàm cho ví dụ này
            {"user_input": "Có loại thuốc uống nào trị mụn nhanh không?", "image": None},
            {"user_input": "Triệu chứng của thủy đậu và sởi khác nhau như thế nào?", "image": None},
            {"user_input": "Da bị lichen hóa là như thế nào?", "image": None},
            {"user_input": "Bạn có thể kể chuyện cười được không?", "image": None},
            {"user_input": "Mẹ tôi bị nổi các mảng đỏ và rất ngứa ở lưng, đó là bệnh gì?", "image": None},
            {"user_input": "Làm thế nào để có một làn da khỏe mạnh và ít bị bệnh?", "image": None},
        ],
        outputs=[
            # Outputs tương ứng với 2 case cũ
            {
                "reference": "Dựa trên mô tả và hình ảnh, đây có thể là triệu chứng của bệnh chàm (eczema) hoặc viêm da tiếp xúc. Bạn nên giữ vùng da sạch sẽ, khô ráo và tránh gãi. Để có chẩn đoán chính xác, bạn nên đi khám bác sĩ."},
            {
                "reference": "Dựa trên hình ảnh, các dấu hiệu này có thể là của bệnh chàm (eczema). Da có vẻ khô, đỏ và có thể ngứa. Bạn nên đi khám bác sĩ để xác nhận chẩn đoán và nhận tư vấn điều trị."},

            # Outputs tương ứng với 10 test case đầu tiên
            {
                "reference": "Chào bạn, các triệu chứng như sốt nhẹ và nổi ban đỏ có thể là dấu hiệu ban đầu của bệnh thủy đậu, thường xuất hiện trong giai đoạn khởi phát. Các triệu chứng khác có thể bao gồm nhức đầu và mệt mỏi. Các nốt ban sau đó có thể phát triển thành mụn nước gây ngứa. Tuy nhiên, đây chỉ là thông tin tham khảo, bạn nên đưa bé đến gặp bác sĩ để được chẩn đoán chính xác."},
            {
                "reference": "Chào bạn, theo thông tin y tế, biện pháp phòng ngừa bệnh thủy đậu hiệu quả và lâu dài nhất là tiêm vắc-xin. Đặc biệt với trẻ em, việc tiêm ngừa vắc-xin thủy đậu là rất quan trọng. Bạn nên tham khảo ý kiến bác sĩ tại các cơ sở y tế để được tư vấn về lịch tiêm chủng phù hợp."},
            {
                "reference": "Chào bạn, viêm mô tế bào thường không lây nhiễm trực tiếp từ người sang người qua đường hô hấp hay tiếp xúc thông thường. Tuy nhiên, bệnh có thể lây nếu vùng da bị đứt hở của bạn tiếp xúc trực tiếp với vùng da bị nhiễm trùng của người bệnh. Để đảm bảo an toàn, bạn nên đi khám bác sĩ để được tư vấn cụ thể."},
            {
                "reference": "Chào bạn, việc điều trị viêm mô tế bào chủ yếu dựa vào kháng sinh theo chỉ định của bác sĩ. Với các trường hợp nhẹ, bác sĩ có thể kê kháng sinh đường uống. Trong các trường hợp nặng hơn hoặc có biến chứng, có thể cần nhập viện để điều trị bằng kháng sinh đường tĩnh mạch. Bạn tuyệt đối không nên tự ý dùng thuốc và cần đi khám để có phác đồ điều trị phù hợp."},
            {
                "reference": "Chào bạn, nguyên nhân gây bệnh chàm da khá phức tạp, thường là sự kết hợp giữa yếu tố di truyền (cơ địa) và các yếu tố từ môi trường. Những người có cơ địa dễ dị ứng khi tiếp xúc với các tác nhân như thực phẩm (trứng, sữa), lông động vật, hóa chất, hoặc thậm chí là căng thẳng, có thể bùng phát bệnh. Để biết rõ hơn, bạn nên tham khảo ý kiến bác sĩ chuyên khoa."},
            {
                "reference": "Chào bạn, các triệu chứng bạn mô tả như da khô, tróc vảy và ngứa là những biểu hiện rất thường gặp của bệnh chàm (eczema). Ngoài ra, bệnh còn có thể có các mảng da đỏ hoặc mụn nước. Tuy nhiên, nhiều bệnh da liễu khác cũng có triệu chứng tương tự, vì vậy bạn nên đi khám bác sĩ để được chẩn đoán chính xác và có hướng điều trị đúng đắn."},
            {
                "reference": "Chào bạn, mề đay là tình trạng phản ứng của da, biểu hiện bằng các nốt mẩn đỏ, sần và ngứa. Bệnh này không lây nhiễm. Hầu hết các trường hợp mề đay cấp tính không nguy hiểm đến tính mạng nhưng gây khó chịu và ảnh hưởng nhiều đến sinh hoạt. Tuy nhiên, trong một số trường hợp nặng, mề đay có thể là dấu hiệu của sốc phản vệ, cần được cấp cứu kịp thời. Bạn nên đi khám bác sĩ để xác định tình trạng của mình."},
            {
                "reference": "Chào bạn, để giảm ngứa do mề đay tại nhà, bạn có thể thử một số cách như chườm lạnh lên vùng da bị ngứa (khoảng 10 phút mỗi lần), hoặc tắm với các dung dịch dịu nhẹ như bột yến mạch. Quan trọng là tránh gãi để không làm tổn thương da và tránh các tác nhân nghi ngờ gây dị ứng. Các biện pháp này chỉ hỗ trợ tạm thời, bạn vẫn nên đi khám bác sĩ để tìm ra nguyên nhân và cách điều trị triệt để."},
            {
                "reference": "Chào bạn, đây là một câu hỏi rất hay. Về cơ bản, viêm da tiếp xúc kích ứng (chiếm khoảng 80%) là phản ứng trực tiếp của da với một chất gây hại như hóa chất mạnh, xảy ra ở hầu hết mọi người. Trong khi đó, viêm da tiếp xúc dị ứng (chiếm 20%) là phản ứng của hệ miễn dịch với một chất mà cơ thể bạn đã bị mẫn cảm từ trước, dù chất đó có thể vô hại với người khác. Để chẩn đoán chính xác, bạn cần đến gặp bác sĩ da liễu."},
            {
                "reference": "Chào bạn, đây là một điều rất không nên làm. Mỗi bệnh da liễu có thể có nguyên nhân và cách điều trị khác nhau, ngay cả khi triệu chứng trông có vẻ giống nhau. Việc dùng chung thuốc bôi không chỉ có thể không hiệu quả mà còn có nguy cơ gây kích ứng hoặc làm tình trạng bệnh của bạn nặng hơn. Bạn nên đi khám bác sĩ để được chẩn đoán và kê đơn thuốc phù hợp với mình."},

            # Outputs tương ứng với 30 test case mới
            {
                "reference": "Chào bạn, thủy đậu có thể để lại sẹo rỗ (lõm) sau khi các nốt mụn nước biến mất. Để hạn chế sẹo, điều quan trọng nhất là không cào, gãi hay làm vỡ các nốt mụn. Việc vệ sinh da sạch sẽ và sử dụng các loại thuốc bôi kháng viêm theo chỉ định của bác sĩ cũng giúp ngăn ngừa nhiễm trùng và giảm nguy cơ sẹo. Bạn nên đi khám để được tư vấn cụ thể."},
            {
                "reference": "Chào bạn, người lớn khi mắc thủy đậu thường có xu hướng bị nặng hơn và gặp nhiều biến chứng nguy hiểm hơn trẻ em, chẳng hạn như viêm phổi. Do đó, nếu là người lớn mắc bệnh, bạn càng cần phải đi khám bác sĩ sớm để được theo dõi và điều trị kịp thời."},
            {
                "reference": "Chào bạn, bệnh thủy đậu rất dễ lây. Người bệnh cần được cách ly từ lúc bắt đầu phát ban cho đến khi tất cả các nốt mụn nước đã khô lại và bong vảy hoàn toàn, thường kéo dài khoảng 7-10 ngày. Bạn nên tham khảo ý kiến bác sĩ để có hướng dẫn cách ly an toàn và chính xác nhất."},
            {
                "reference": "Chào bạn, có, viêm mô tế bào có thể xảy ra ở bất kỳ đâu trên cơ thể, bao gồm cả vùng mặt. Viêm mô tế bào quanh mắt là một tình trạng đặc biệt nguy hiểm, có thể gây biến chứng nghiêm trọng đến thị lực. Nếu có dấu hiệu sưng đỏ ở mặt, bạn cần đi khám bác sĩ ngay lập tức."},
            {
                "reference": "Chào bạn, nếu không được điều trị kịp thời bằng kháng sinh, viêm mô tế bào có thể lan rộng và gây ra các biến chứng rất nguy hiểm như áp xe dưới da, viêm cân cơ hoại tử, hoặc nhiễm khuẩn huyết, có thể đe dọa đến tính mạng. Do đó, việc đi khám sớm là cực kỳ quan trọng."},
            {
                "reference": "Chào bạn, bệnh chàm thường trở nên nặng hơn vào mùa đông vì không khí lạnh và khô làm da mất đi độ ẩm tự nhiên, trở nên khô hơn và dễ bị kích ứng. Việc sử dụng máy sưởi cũng làm giảm độ ẩm trong không khí. Bạn nên tăng cường dưỡng ẩm trong mùa này và đi khám bác sĩ để được tư vấn."},
            {
                "reference": "Chào bạn, người bị chàm nên sử dụng các loại sữa tắm và xà phòng rất dịu nhẹ, không chứa hương liệu, thuốc nhuộm hay chất tẩy rửa mạnh. Tắm bằng nước ấm (không quá nóng) và thoa kem dưỡng ẩm ngay sau khi lau khô người là rất quan trọng. Bạn nên tham khảo ý kiến bác sĩ để chọn sản phẩm phù hợp."},
            {
                "reference": "Chào bạn, chàm (viêm da cơ địa) rất phổ biến ở trẻ sơ sinh và trẻ nhỏ. Nhiều trường hợp bệnh có thể tự cải thiện khi trẻ lớn lên, nhưng một số trường hợp có thể kéo dài. Điều quan trọng là chăm sóc da đúng cách và điều trị theo chỉ dẫn của bác sĩ để kiểm soát triệu chứng và giúp bé dễ chịu."},
            {
                "reference": "Chào bạn, ở một số người, thực phẩm có thể là yếu tố làm bùng phát bệnh chàm, các loại thường gặp là trứng, sữa, hải sản. Tuy nhiên, việc này không giống nhau ở tất cả mọi người. Bạn không nên tự ý kiêng khem quá mức mà nên đi khám bác sĩ để được tư vấn và có thể làm các xét nghiệm dị ứng nếu cần thiết."},
            {
                "reference": "Chào bạn, căng thẳng (stress) là một trong những yếu tố đã được chứng minh có thể kích hoạt hoặc làm nặng thêm tình trạng nổi mề đay. Việc kiểm soát căng thẳng thông qua các phương pháp như thiền, yoga, hoặc tập thể dục có thể hữu ích. Tuy nhiên, bạn vẫn nên đi khám bác sĩ để loại trừ các nguyên nhân khác và có hướng điều trị phù hợp."},
            {
                "reference": "Chào bạn, một đợt mề đay cấp tính thường kéo dài không quá 24 giờ tại một vị trí, mặc dù các nốt mới có thể tiếp tục xuất hiện ở nơi khác. Toàn bộ đợt bệnh cấp tính thường kết thúc trong vòng dưới 6 tuần. Nếu tình trạng kéo dài hơn 6 tuần, nó được coi là mề đay mãn tính và bạn cần đi khám bác sĩ."},
            {
                "reference": "Chào bạn, mề đay thường có đặc điểm là các nốt sẩn, phù nề, có giới hạn rõ và rất ngứa, chúng có thể thay đổi vị trí nhanh chóng. Các loại phát ban khác có thể có biểu hiện đa dạng hơn như mụn nước, vảy da, và thường không biến mất nhanh như mề đay. Để phân biệt chính xác, bạn cần được bác sĩ thăm khám trực tiếp."},
            {
                "reference": "Chào bạn, nổi mề đay trong thai kỳ khá phổ biến do sự thay đổi nội tiết tố. Hầu hết các trường hợp không gây hại cho em bé, nhưng gây khó chịu cho mẹ. Tuy nhiên, việc sử dụng bất kỳ loại thuốc nào trong thai kỳ đều cần có sự chỉ định và theo dõi chặt chẽ của bác sĩ. Bạn nên đi khám để được tư vấn an toàn."},
            {
                "reference": "Chào bạn, đây là một kịch bản rất điển hình của viêm da tiếp xúc dị ứng. Niken là một kim loại thường có trong trang sức không nguyên chất và là một trong những dị nguyên phổ biến nhất. Bạn nên ngưng đeo món trang sức đó và đi khám bác sĩ để được chẩn đoán và điều trị đúng cách."},
            {
                "reference": "Chào bạn, hoàn toàn có khả năng bạn đã bị viêm da tiếp xúc do một loại thực vật nào đó. Một số cây chứa nhựa hoặc chất gây dị ứng có thể gây ra phản ứng da như ngứa và nổi mụn nước khi tiếp xúc. Bạn nên đi khám bác sĩ để được xác định nguyên nhân và có thuốc bôi phù hợp."},
            {
                "reference": "Chào bạn, thời gian để viêm da tiếp xúc khỏi bệnh phụ thuộc vào việc bạn có loại bỏ hoàn toàn tác nhân gây bệnh hay không và mức độ nặng của tổn thương. Nếu loại bỏ được nguyên nhân và điều trị đúng cách, các triệu chứng cấp tính có thể cải thiện sau vài ngày đến 2 tuần. Bạn cần tuân thủ chỉ định của bác sĩ."},
            {
                "reference": "Chào bạn, đúng vậy. Bệnh zona (giời leo) và bệnh thủy đậu đều do cùng một loại virus gây ra, đó là Varicella-zoster virus. Sau khi một người đã bị thủy đậu, virus này không biến mất mà tồn tại ở trạng thái không hoạt động trong các dây thần kinh và có thể tái hoạt động sau này để gây ra bệnh zona. Bạn nên đi khám bác sĩ nếu nghi ngờ mắc bệnh."},
            {
                "reference": "Chào bạn, bệnh vảy nến là một bệnh da liễu mạn tính, có nghĩa là hiện tại chưa có phương pháp chữa trị dứt điểm hoàn toàn. Tuy nhiên, có rất nhiều phương pháp điều trị hiệu quả giúp kiểm soát tốt các triệu chứng, giữ cho bệnh ổn định và không ảnh hưởng đến chất lượng cuộc sống. Bạn cần được bác sĩ chuyên khoa theo dõi và điều trị lâu dài."},
            {
                "reference": "Chào bạn, về cơ bản, 'viêm da cơ địa' là thuật ngữ y khoa chính xác và phổ biến nhất để chỉ một dạng chàm (eczema) nội sinh, liên quan đến yếu tố di truyền và cơ địa dị ứng. Trong giao tiếp thông thường, nhiều người dùng hai thuật ngữ này thay thế cho nhau. Bạn nên đi khám bác sĩ để được chẩn đoán chính xác tình trạng của mình."},
            {
                "reference": "Chào bạn, việc phân biệt nấm da và chàm đôi khi khá khó khăn vì triệu chứng có thể tương tự nhau (đỏ, ngứa, bong vảy). Tuy nhiên, tổn thương do nấm thường có bờ viền rõ, hình tròn hoặc đa cung, trong khi chàm thường có giới hạn không rõ. Để chẩn đoán chính xác, bác sĩ cần thăm khám và có thể sẽ phải cạo vảy da để soi nấm dưới kính hiển vi."},
            {
                "reference": "Chào bạn, tôi hiểu sự lo lắng của bạn. Nếu bạn đã sử dụng thuốc theo đúng chỉ định của bác sĩ mà triệu chứng vẫn không cải thiện, điều quan trọng là bạn nên liên hệ lại hoặc tái khám với bác sĩ đó. Có thể bạn cần điều chỉnh liều lượng hoặc đổi sang một phác đồ điều trị khác. Bạn không nên tự ý ngưng thuốc hoặc đổi thuốc."},
            {
                "reference": "Chào bạn, hiện tại không có bằng chứng khoa học đáng tin cậy nào cho thấy việc dùng lá trầu không có thể chữa khỏi bệnh chàm. Thậm chí, một số phương pháp dân gian có thể gây kích ứng hoặc nhiễm trùng da, làm tình trạng bệnh nặng hơn. Bạn nên tuân thủ các phương pháp điều trị đã được y học chứng minh và tham khảo ý kiến bác sĩ."},
            {
                "reference": "Chào bạn, tôi rất sẵn lòng lắng nghe. Bạn có thể mô tả cụ thể hơn về vấn đề da bạn đang gặp phải không? Ví dụ như da có biểu hiện gì (đỏ, ngứa, khô, nổi mụn...), ở vị trí nào và đã kéo dài bao lâu rồi? Càng có nhiều thông tin, tôi càng có thể hỗ trợ bạn tốt hơn. Tuy nhiên, lời khuyên tốt nhất vẫn là đi khám bác sĩ chuyên khoa."},
            {
                "reference": "Chào bạn, cảm ơn bạn đã cung cấp hình ảnh. Tuy nhiên, việc chẩn đoán bệnh da liễu đòi hỏi phải quan sát trực tiếp tổn thương một cách rõ ràng và kết hợp với việc hỏi bệnh. Qua hình ảnh này, tôi không thể đưa ra kết luận chính xác. Bạn vui lòng đến gặp bác sĩ da liễu để được thăm khám và chẩn đoán đúng bệnh."},
            {
                "reference": "Chào bạn, có nhiều loại thuốc uống được sử dụng để điều trị mụn, nhưng tất cả đều cần được kê đơn và theo dõi bởi bác sĩ chuyên khoa da liễu. Việc tự ý sử dụng thuốc uống trị mụn có thể gây ra nhiều tác dụng phụ nghiêm trọng. Bạn nên đi khám để bác sĩ đánh giá tình trạng mụn và chỉ định loại thuốc phù hợp và an toàn."},
            {
                "reference": "Chào bạn, mặc dù cả thủy đậu và sởi đều gây sốt và phát ban, nhưng chúng có những điểm khác biệt. Ban của sởi thường là ban dạng sẩn, phẳng, trong khi ban của thủy đậu nhanh chóng phát triển thành các mụn nước chứa dịch trong. Ngoài ra, sởi thường có các triệu chứng viêm đường hô hấp (ho, chảy nước mũi) và các dấu hiệu đặc trưng khác. Bạn nên đưa bệnh nhân đi khám bác sĩ để có chẩn đoán chính xác."},
            {
                "reference": "Chào bạn, 'lichen hóa' là một thuật ngữ y khoa để mô tả tình trạng da trở nên dày, cứng và các nếp da hằn sâu hơn bình thường, trông giống như vỏ cây. Đây là hậu quả của việc cào, gãi hoặc chà xát da một cách mạn tính, thường gặp ở những người bị bệnh chàm (viêm da cơ địa) lâu ngày. Bạn nên đi khám bác sĩ để được điều trị."},
            {
                "reference": "Tôi là một trợ lý ảo chuyên về các vấn đề y tế da liễu. Rất tiếc tôi không có khả năng kể chuyện cười. Bạn có cần tôi hỗ trợ thông tin nào về chăm sóc da hoặc các bệnh về da không?"},
            {
                "reference": "Chào bạn, các triệu chứng như nổi mảng đỏ và ngứa ở lưng có thể do nhiều nguyên nhân gây ra, chẳng hạn như mề đay, chàm, hoặc viêm da tiếp xúc. Để biết chính xác đó là bệnh gì và có cách điều trị phù hợp, mẹ của bạn nên đi khám trực tiếp với bác sĩ chuyên khoa da liễu."},
            {
                "reference": "Chào bạn, để có một làn da khỏe mạnh, bạn nên duy trì một chế độ chăm sóc da cơ bản bao gồm làm sạch, dưỡng ẩm và chống nắng hàng ngày. Ngoài ra, việc uống đủ nước, có chế độ ăn uống cân bằng, ngủ đủ giấc và kiểm soát căng thẳng cũng đóng vai trò rất quan trọng. Nếu có vấn đề cụ thể, bạn nên tham khảo ý kiến bác sĩ da liễu."},
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
    dataset_name = "Derm Assistant Test Cases - Final"


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
    # judge_llm = ChatOpenAI(
    #     temperature=0,
    #     api_key=os.getenv("OPENAI_KEY"),
    #     model="meta-llama/llama-4-maverick:free",
    #     base_url="https://openrouter.ai/api/v1",
    #     max_tokens=3072,
    # )

    judge_llm = ChatGroq(
        temperature=0,
        api_key=GROQ_API_KEY,
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
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

    async def run_graph_with_dataset_inputs_with_delay(input_dict: dict):
        # Chờ 3 giây trước mỗi lần chạy để đảm bảo không vượt rate limit
        await asyncio.sleep(6)
        return await run_graph_with_dataset_inputs(input_dict)

    # # --- Chạy evaluate với evaluator mới ---
    # results = await aevaluate(
    #     run_graph_with_dataset_inputs,
    #     data=dataset_name,
    #     evaluators=[
    #         run_eval_help,
    #         run_eval_coh,
    #         run_eval_correct,
    #         run_eval_hallucination  # MỚI
    #     ],
    #     experiment_prefix="Derm Assistant Run On Dataset - Final",
    #     max_concurrency=1
    # )
    results = await aevaluate(
        # Sử dụng hàm có độ trễ ở đây
        run_graph_with_dataset_inputs_with_delay,
        data=dataset_name,
        evaluators=[run_eval_hallucination ,run_eval_correct],
        experiment_prefix="...",
        # Vẫn có thể kết hợp với max_concurrency
        max_concurrency=1,
    )

    print("\n✅ Test hoàn thành. Experiment:", results.experiment_name)
    print("Mở LangSmith UI để xem chi tiết.")


if __name__ == "__main__":
    asyncio.run(main())