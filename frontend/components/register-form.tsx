import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {FormEvent} from "react";

export function RegisterForm({
  className,
  onSubmit,
  onSwitchToLogin,
  ...props
}: React.ComponentProps<"form"> & {
  onSubmit?: (event: FormEvent<HTMLFormElement>) => void | Promise<void>;
  onSwitchToLogin?: () => void;
}) {
  return (
    <form className={cn("flex flex-col gap-6", className)}
          {...props}
          onSubmit={onSubmit}>
      <div className="flex flex-col items-center gap-2 text-center">
        <h1 className="text-2xl font-bold">Tạo tài khoản mới</h1>
        <p className="text-muted-foreground text-sm text-balance">
          Nhập thông tin để tạo tài khoản của bạn
        </p>
      </div>
      <div className="grid gap-6">
        <div className="grid gap-3">
          <Label htmlFor="register-email">Email</Label>
          <Input
            id="register-email"
            type="email"
            name="email"
            placeholder="m@example.com"
            required
          />
        </div>
        <div className="grid gap-3">
          <Label htmlFor="register-password">Mật khẩu</Label>
          <Input
            id="register-password"
            type="password"
            name="password"
            placeholder="Tối thiểu 6 ký tự"
            minLength={6}
            required
          />
        </div>
        <div className="grid gap-3">
          <Label htmlFor="confirm-password">Xác nhận mật khẩu</Label>
          <Input
            id="confirm-password"
            type="password"
            name="confirmPassword"
            placeholder="Nhập lại mật khẩu"
            minLength={6}
            required
          />
        </div>
        <Button type="submit" className="w-full">
           Đăng ký
        </Button>

      </div>
      <div className="text-center text-sm">
        Đã có tài khoản?{" "}
        <button
          type="button"
          onClick={onSwitchToLogin}
          className="underline underline-offset-4 hover:text-primary"
        >
          Đăng nhập
        </button>
      </div>
    </form>
  )
}