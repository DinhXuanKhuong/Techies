export async function POST(req: Request) {
  const { messages } = await req.json();

  // chỉ lấy câu hỏi cuối cùng
  const userMessage = messages[messages.length - 1].content;

  const response = await fetch(`http://localhost:8000/chat?q=${encodeURIComponent(userMessage)}`, {
    method: "GET",
  });

  return new Response(response.body, {
    headers: { "Content-Type": "text/plain" },
  });
}