"use client"
import { GalleryVerticalEnd } from "lucide-react";
import {supabase} from "@/lib/supabase";
import { LoginForm } from "@/components/login-form"
import { RegisterForm } from "@/components/register-form"
import {useRouter} from "next/navigation";
import {FormEvent, useState} from "react";

export default function HomePage() {
  const router = useRouter();
  const [isLogin, setIsLogin] = useState(true);

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const email = formData.get('email') as string;
    const password = formData.get('password') as string;

    const { error } = await supabase.auth.signInWithPassword({email, password});

    if (error) {
      alert(error.message)
    } else {
      router.push("/chat");
    }
  }

  async function handleRegister(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const email = formData.get('email') as string;
    const password = formData.get('password') as string;
    const confirmPassword = formData.get('confirmPassword') as string;

    if (password !== confirmPassword) {
      alert("Mật khẩu xác nhận không khớp!");
      return;
    }

    if (password.length < 6) {
      alert("Mật khẩu phải có ít nhất 6 ký tự!");
      return;
    }

    const { error } = await supabase.auth.signUp({
      email,
      password,
    });

    if (error) {
      alert(error.message);
    } else {
      alert("Đăng ký thành công! Vui lòng kiểm tra email để xác nhận tài khoản.");
      setIsLogin(true); // Chuyển về form login
    }
  }

  return (
    <div className="grid min-h-svh lg:grid-cols-2">
      <div className="flex flex-col gap-4 p-6 md:p-10">
        <div className="flex justify-center gap-2 md:justify-start">
          <a href="#" className="flex items-center gap-2 font-medium">
            <div className="bg-primary text-primary-foreground flex size-6 items-center justify-center rounded-md">
              <GalleryVerticalEnd className="size-4" />
            </div>
            DefmAI
          </a>
        </div>
        <div className="flex flex-1 items-center justify-center">
          <div className="w-full max-w-xs">
            {isLogin ? (
              <LoginForm 
                onSubmit={handleLogin} 
                onSwitchToRegister={() => setIsLogin(false)}
              />
            ) : (
              <RegisterForm 
                onSubmit={handleRegister}
                onSwitchToLogin={() => setIsLogin(true)}
              />
            )}
          </div>
        </div>
      </div>
      <div className="bg-muted relative hidden lg:block">
        <img
          src="https://qezsedgptmntyjrfkqbl.supabase.co/storage/v1/object/public/chat-files/chat_uploads/16334676_Tiny%20dermatologists%20examining%20skin%20of%20patient%20at%20hospital.jpg"
          alt="Image"
          className="absolute inset-0 h-full w-full object-cover dark:brightness-[0.2] dark:grayscale"
        />
      </div>
    </div>
  )
}