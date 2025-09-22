import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {FormEvent} from "react";

export function LoginForm({
  className,
  onSubmit,
  onSwitchToRegister,
  ...props
}: React.ComponentProps<"form"> & {
  onSubmit?: (event: FormEvent<HTMLFormElement>) => void | Promise<void>;
  onSwitchToRegister?: () => void;
}) {
  return (
    <form className={cn("flex flex-col gap-6", className)}
          {...props}
          onSubmit={onSubmit}>
      <div className="flex flex-col items-center gap-2 text-center">
        <h1 className="text-2xl font-bold">Đăng nhập vào tài khoản của bạn</h1>
        <p className="text-muted-foreground text-sm text-balance">
          Nhập email và password để vào tài khoản
        </p>
      </div>
      <div className="grid gap-6">
        <div className="grid gap-3">
          <Label htmlFor="email">Email</Label>
          <Input id="email" type="email" name="email" placeholder="m@example.com" required />
        </div>
        <div className="grid gap-3">
          <div className="flex items-center">
            <Label htmlFor="password">Password</Label>
            <a
              href="#"
              className="ml-auto text-sm underline-offset-4 hover:underline"
            >
              Quên mật khẩu?
            </a>
          </div>
          <Input id="password" type="password" name="password" required />
        </div>
        <Button type="submit" className="w-full">
           Đăng nhập
        </Button>

      </div>
      <div className="text-center text-sm">
        Chưa có tài khoản?{" "}
        <button
          type="button"
          onClick={onSwitchToRegister}
          className="underline underline-offset-4 hover:text-primary"
        >
          Đăng kí
        </button>
      </div>
    </form>
  )
}
