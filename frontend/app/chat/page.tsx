"use client";

import { useEffect, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { supabase } from "@/lib/supabase";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import {
  Send,
  Bot,
  User,
  Loader2,
  MessageSquarePlus,
  MoreVertical,
  Trash2,
  Edit,
  Menu,
  ImagePlus,
  X
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ScrollArea } from "@/components/ui/scroll-area";

// type Msg = { role: string; content: string; created_at?: string };
type Msg = {
  role: string;
  content: string;
  image_url?: string; // thêm cái này
  created_at?: string;
};

type ChatSession = {
  session_id: string;
  title: string;
  updated_at: string;
  message_count: number;
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<string>(() => uuidv4());
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null); // Ref cho file input
  const [imagePreview, setImagePreview] = useState<string | null>(null); // Thêm preview

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load sessions list
  useEffect(() => {
    loadSessions();
  }, []);

  // Load messages when session changes
  useEffect(() => {
    if (currentSessionId) {
      fetchHistory(currentSessionId);
    }
  }, [currentSessionId]);

    // Xử lý khi chọn file
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        alert("File quá lớn. Vui lòng chọn file nhỏ hơn 10MB.");
        return;
      }

      if (!file.type.startsWith('image/')) {
        alert("Vui lòng chọn file ảnh.");
        return;
      }

      setSelectedFile(file);

      // Tạo preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // Xóa file đã chọn
  const removeSelectedFile = () => {
    setSelectedFile(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const loadSessions = async () => {
    const { data: { session } } = await supabase.auth.getSession();
    if (!session) return;

    try {
      const res = await fetch(`http://localhost:8000/sessions`, {
        headers: { Authorization: `Bearer ${session.access_token}` },
      });

      if (res.ok) {
        const sessionsList: ChatSession[] = await res.json();
        setSessions(sessionsList);
      }
    } catch (error) {
      console.error("Failed to load sessions", error);
    }
  };

  const fetchHistory = async (sessionId: string) => {
    const { data: { session } } = await supabase.auth.getSession();
    if (!session) return;

    try {
      const res = await fetch(`http://localhost:8000/history?session_id=${sessionId}`, {
        headers: { Authorization: `Bearer ${session.access_token}` },
      });

      if (res.ok) {
        const history: Msg[] = await res.json();
        setMessages(history.map(h => ({
          role: h.role,
          content: h.content,
          image_url: h.image_url,
          created_at: h.created_at
        })));
      } else {
        console.error("Failed to load history", await res.text());
      }
    } catch (error) {
      console.error("Error fetching history:", error);
      setMessages([]);
    }
  };

  const createNewChat = async () => {
    const { data: { session } } = await supabase.auth.getSession();
    if (!session) return;

    try {
      const res = await fetch(`http://localhost:8000/sessions`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${session.access_token}` },
      });

      if (res.ok) {
        const newSession = await res.json();
        setCurrentSessionId(newSession.session_id);
        setMessages([]);
        loadSessions(); // Refresh sessions list

        removeSelectedFile(); // Clear selected file when creating new chat
      }
    } catch (error) {
      console.error("Failed to create new session", error);
    }
  };

  const deleteSession = async (sessionId: string) => {
    const { data: { session } } = await supabase.auth.getSession();
    if (!session) return;

    if (!confirm("Bạn có chắc muốn xóa cuộc trò chuyện này?")) return;

    try {
      const res = await fetch(`http://localhost:8000/sessions/${sessionId}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${session.access_token}` },
      });

      if (res.ok) {
        // If deleting current session, create a new one
        if (sessionId === currentSessionId) {
          await createNewChat();
        }
        loadSessions(); // Refresh sessions list
      }
    } catch (error) {
      console.error("Failed to delete session", error);
    }
  };

  const updateSessionTitle = async (sessionId: string, newTitle: string) => {
    const { data: { session } } = await supabase.auth.getSession();
    if (!session) return;

    try {
      const res = await fetch(`http://localhost:8000/sessions/${sessionId}`, {
        method: 'PUT',
        headers: {
          Authorization: `Bearer ${session.access_token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ title: newTitle })
      });

      if (res.ok) {
        loadSessions(); // Refresh sessions list
      }
    } catch (error) {
      console.error("Failed to update session title", error);
    }
  };

  const handleEditTitle = (sessionId: string, currentTitle: string) => {
    setEditingSessionId(sessionId);
    setEditingTitle(currentTitle);
  };

  const handleSaveTitle = async (sessionId: string) => {
    if (editingTitle.trim()) {
      await updateSessionTitle(sessionId, editingTitle.trim());
    }
    setEditingSessionId(null);
    setEditingTitle("");
  };


  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() && !selectedFile) return;

    const { data: { session } } = await supabase.auth.getSession();
    if (!session) {
      alert("Bạn cần đăng nhập");
      return;
    }
    const token = session.access_token;

    // Tạo message preview cho user
    let userMessage = input.trim();
    if (selectedFile && !userMessage) {
      userMessage = `[Đã gửi ảnh: ${selectedFile.name}]`;
    } else if (selectedFile && userMessage) {
      userMessage = `${userMessage} [Kèm ảnh: ${selectedFile.name}]`;
    }

    setMessages(prev => [...prev, {
      role: "user",
      content: userMessage,
      image_url: imagePreview || undefined // Hiển thị preview trong chat
    }]);

    const formData = new FormData();
    formData.append("q", input || "");
    formData.append("session_id", currentSessionId);
    if (selectedFile) formData.append("file", selectedFile);

    setInput("");
    removeSelectedFile(); // Clear file sau khi gửi
    setLoading(true);

    try {
      const res = await fetch(`http://localhost:8000/chat`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });

      if (!res.body) {
        setLoading(false);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      setMessages(prev => [...prev, { role: "assistant", content: "" }]);

      let assistantContent = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        assistantContent += chunk;
        setMessages(prev => {
          const copy = [...prev];
          copy[copy.length - 1] = { role: "assistant", content: assistantContent };
          return copy;
        });
      }

      setLoading(false);
      loadSessions();
    } catch (error) {
      console.error("Error sending message:", error);
      setLoading(false);
    }
  }
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60);

    if (diffInHours < 24) {
      return date.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
    } else if (diffInHours < 24 * 7) {
      return date.toLocaleDateString("vi-VN", { weekday: "short" });
    } else {
      return date.toLocaleDateString("vi-VN", { month: "short", day: "numeric" });
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-slate-900">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-80' : 'w-0'} transition-all duration-300 overflow-hidden bg-white dark:bg-slate-800 border-r border-slate-200 dark:border-slate-700 flex flex-col`}>
        {/* Sidebar Header */}
        <div className="p-4 border-b border-slate-200 dark:border-slate-700">
          <Button
            onClick={createNewChat}
            className="w-full justify-start gap-2 bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300"
          >
            <MessageSquarePlus className="w-4 h-4" />
            Cuộc trò chuyện mới
          </Button>
        </div>

        {/* Sessions List */}
        <ScrollArea className="flex-1">
          <div className="p-2 space-y-1">
            {sessions.map((session) => (
              <div
                key={session.session_id}
                className={`group relative flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors ${
                  session.session_id === currentSessionId
                    ? "bg-slate-200 dark:bg-slate-700"
                    : "hover:bg-slate-100 dark:hover:bg-slate-700/50"
                }`}
                onClick={() => setCurrentSessionId(session.session_id)}
              >
                <div className="flex-1 min-w-0">
                  {editingSessionId === session.session_id ? (
                    <Input
                      value={editingTitle}
                      onChange={(e) => setEditingTitle(e.target.value)}
                      onBlur={() => handleSaveTitle(session.session_id)}
                      onKeyPress={(e) => {
                        if (e.key === 'Enter') {
                          handleSaveTitle(session.session_id);
                        }
                      }}
                      className="h-6 text-sm"
                      autoFocus
                    />
                  ) : (
                    <>
                      <div className="text-sm font-medium text-slate-800 dark:text-slate-200 truncate">
                        {session.title}
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                        <span>{formatDate(session.updated_at)}</span>
                        <span>•</span>
                        <span>{session.message_count} tin nhắn</span>
                      </div>
                    </>
                  )}
                </div>

                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="opacity-0 group-hover:opacity-100 transition-opacity h-6 w-6 p-0"
                    >
                      <MoreVertical className="w-3 h-3" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => handleEditTitle(session.session_id, session.title)}>
                      <Edit className="w-4 h-4 mr-2" />
                      Đổi tên
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onClick={() => deleteSession(session.session_id)}
                      className="text-red-600 dark:text-red-400"
                    >
                      <Trash2 className="w-4 h-4 mr-2" />
                      Xóa
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Top Bar */}
        <div className="flex-none bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg border-b border-slate-200/50 dark:border-slate-700/50 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="lg:hidden"
              >
                {sidebarOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
              </Button>


            </div>

            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="hidden lg:flex"
            >
              {sidebarOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
            </Button>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.length === 0 && (
              <div className="text-center py-12">
                {/*<div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30 mb-4">*/}
                {/*  <Bot className="w-8 h-8 text-blue-600 dark:text-blue-400" />*/}
                {/*</div>*/}
                  <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-gradient-to-r from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30 mb-4 overflow-hidden">
                      <img
                        src="https://qezsedgptmntyjrfkqbl.supabase.co/storage/v1/object/public/chat-files/chat_uploads/551884845_6072882859625357_7744008671031620226_n.png"
                        alt="Logo"
                        className="w-14 h-14  object-contain"
                      />
                  </div>
                <h3 className="text-lg font-medium text-slate-800 dark:text-slate-200 mb-2">
                  Chào bạn👋! Tôi là một chatbot chuyên về da liễu, tôi có thể chẩn đoán bệnh và đưa ra lời khuyên phù hợp với bạn
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                  Hãy bắt đầu cuộc trò chuyện bằng cách gửi tin nhắn của bạn. Lưu ý: Câu trả lời của chatbot chỉ để tham khảo và không có tác dụng thay thế lời khuyên của bác sĩ và chuyên gia.
                </p>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex gap-4 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >

                {msg.role === "assistant" && (
                  <div className="flex-none flex items-start">
                    <div className="flex items-center justify-center w-12 h-12 rounded-full bg-gradient-to-r from-blue-500 to-purple-600">
                      <Bot className="w-8 h-8 text-white" />
                    </div>
                  </div>
                )}

                <div
                  className={`group relative max-w-[75%] ${
                    msg.role === "user"
                      ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg shadow-blue-500/25"
                      : "bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 shadow-lg shadow-slate-500/10 dark:shadow-slate-900/20 border border-slate-200/50 dark:border-slate-700/50"
                  } rounded-2xl px-5 py-3 backdrop-blur-sm transition-all duration-200 hover:shadow-xl`}
                >
                    {msg.image_url && (
                    <img
                      src={msg.image_url}
                      alt="Ảnh đính kèm"
                      className="mb-3 max-w-full max-h-64 rounded-lg object-cover border border-slate-200/50 dark:border-slate-700/50"
                    />
                  )}
                  <MarkdownRenderer
                    content={msg.content}
                    className={msg.role === "user" ? "text-white" : ""}
                  />

                  {msg.created_at && (
                    <div
                      className={`text-xs mt-2 opacity-0 group-hover:opacity-60 transition-opacity ${
                        msg.role === "user" ? "text-blue-100" : "text-slate-400"
                      }`}
                    >
                      {new Date(msg.created_at).toLocaleTimeString("vi-VN", {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </div>
                  )}
                </div>

                {msg.role === "user" && (
                  <div className="flex-none flex items-start">
                    <div className="flex items-center justify-center w-12 h-12 rounded-full bg-gradient-to-r from-green-500 to-emerald-600">
                      <User className="w-8 h-8 text-white" />
                    </div>
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div className="flex gap-4 justify-start">

                <div className="bg-white dark:bg-slate-800 rounded-2xl px-5 py-3 shadow-lg border border-slate-200/50 dark:border-slate-700/50">
                  <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm">Đang trả lời...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

          {/* Input Area  */}
        <div className="flex-none bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg border-t border-slate-200/50 dark:border-slate-700/50 px-6 py-4">
          <div className="max-w-4xl mx-auto">
            {/* Image Preview */}
            {imagePreview && (
              <div className="mb-3 relative inline-block">
                <img
                  src={imagePreview}
                  alt="Preview"
                  className="max-h-32 max-w-xs rounded-lg border border-slate-200 dark:border-slate-600 object-cover"
                />
                <Button
                  type="button"
                  onClick={removeSelectedFile}
                  size="sm"
                  variant="destructive"
                  className="absolute -top-2 -right-2 h-6 w-6 p-0 rounded-full"
                >
                  <X className="w-4 h-4" />
                </Button>
                <p className="text-xs text-slate-500 mt-1">{selectedFile?.name}</p>
              </div>
            )}

            <form onSubmit={handleSubmit} className="flex gap-3">
              {/* Hidden file input */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />

              {/* Image upload button */}
              <Button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                variant="outline"
                size="sm"
                className="h-12 px-3 rounded-xl border-slate-300 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-700 transition-all duration-200"
                disabled={loading}
              >
                <ImagePlus className="w-5 h-5 text-slate-600 dark:text-slate-400" />
              </Button>

              <div className="relative flex-1">
                <Input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={selectedFile ? "Mô tả ảnh (tùy chọn)..." : "Nhập tin nhắn của bạn..."}
                  className="pr-12 h-12 rounded-xl border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                  disabled={loading}
                />

                <div className="absolute right-3 top-1/2 -translate-y-1/2">
                  <Button
                    type="submit"
                    disabled={loading || (!input.trim() && !selectedFile)}
                    size="sm"
                    className="h-8 w-8 p-0 rounded-lg bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 shadow-md hover:shadow-lg transition-all duration-200"
                  >
                    {loading ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Send className="w-4 h-4" />
                    )}
                  </Button>
                </div>
              </div>
            </form>

            <div className="flex items-center justify-center mt-2">
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {selectedFile
                  ? `Đã chọn: ${selectedFile.name} • Nhấn Enter để gửi`
                  : "Lưu ý: Câu trả lời từ chatbot sẽ không đúng 100% và không thể thay thế lời khuyên của bác sĩ"
                }
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

