use crate::fir::*;

pub struct FIRStringifier<'fir> {
    fir: &'fir FIR,
    str: String,
    cur_func_i: Option<usize>,
    offset: u32,
}

#[allow(dead_code)]
impl<'fir> FIRStringifier<'fir> {
    const INDENT: &'static str = "  ";

    pub fn stringify(fir: &'fir FIR) -> String {
        let this = Self {
            fir,
            str: String::new(),
            cur_func_i: None,
            offset: 0,
        };
        return this._stringify();
    }

    fn _stringify(mut self) -> String {
        for (i, inst) in self.fir.ops.iter().enumerate() {
            match inst {
                Op::Load(r) => {
                    self.inst_eq(i);
                    let ty = &self.fir.types[i];
                    self.op_func_2_ty_ref(inst, ty, r);
                }
                Op::Add(lhs, rhs)
                | Op::Sub(lhs, rhs)
                | Op::Mul(lhs, rhs)
                | Op::Div(lhs, rhs)
                | Op::Gt(lhs, rhs)
                | Op::GtEq(lhs, rhs)
                | Op::Lt(lhs, rhs)
                | Op::LtEq(lhs, rhs)
                | Op::Eq(lhs, rhs) => {
                    self.inst_eq(i);
                    let ty = &self.fir.types[i];
                    self.op_func_3_ty_ref_ref(inst, ty, lhs, rhs);
                }
                Op::FunDef { name } => {
                    assert!(!self.in_func(), "nested functions not supported");
                    self.write("define");
                    self.space();
                    let return_ty = &self.fir.types[i];
                    self.write_type_ref(return_ty);
                    self.space();
                    self.func_ref(*name);
                    self.space();
                    self.write("{");
                    self.cur_func_i = Some(i + 1);
                }
                Op::FunArg => {
                    assert!(self.in_func(), "arg decl outside of function");
                    self.inst_eq(i);
                    let ty = &self.fir.types[i];
                    self.op_func_1_ty(inst, ty);
                }
                Op::FunCall { fun: fun_i, args } => {
                    let Op::FunDef { name } = &self.fir.ops[*fun_i as usize] else {
                        unreachable!("fun call ref not fun def");
                    };
                    self.inst_eq(i);
                    self.op_name(*inst); // "call"
                    self.paren_start();
                    let ret_ty = &self.fir.types[i];
                    self.write_type_ref(ret_ty);
                    self.sep();
                    self.func_ref(*name);
                    self.sep();
                    self.brack_start();
                    let ast::ExtraFunArgs { args } = self.fir.extra.get::<ast::ExtraFunArgs>(*args);
                    for arg in args.iter().take(args.len() - 1) {
                        self.inst_ref(*arg);
                        self.sep();
                    }
                    self.inst_ref(*args.last().unwrap());
                    self.brack_end();
                    self.paren_end();
                }
                Op::Ret(r) => {
                    self.inst_eq(i);
                    self.op_func_1_ref(inst, r);
                }
                Op::FunEnd => {
                    self.write("}");
                }
                Op::Alloc => {
                    self.inst_eq(i);
                    let ty = &self.fir.types[i];
                    self.op_func_1_ty(inst, ty);
                }
                Op::Store(dest, src) => {
                    self.op_func_2_ref_ref(inst, dest, src);
                }
                Op::Branch { cond, t, f } => {
                    self.op_func_3_ref_ref_ref(inst, cond, t, f);
                }
                Op::Jump(dest) => {
                    self.op_func_1_ref(inst, dest);
                }
                Op::Phi {
                    a: (a_from, a_res),
                    b: (b_from, b_res),
                } => {
                    self.inst_eq(i);
                    self.op_name(*inst);
                    self.paren_start();

                    self.write_type_ref_at(i);

                    self.sep();

                    self.brack_start();
                    self.write_ref(a_from);
                    self.write(" -> ");
                    self.write_ref(a_res);
                    self.brack_end();

                    self.sep();

                    self.brack_start();
                    self.write_ref(b_from);
                    self.write(" -> ");
                    self.write_ref(b_res);
                    self.brack_end();

                    self.paren_end();
                }
                Op::Label => {
                    self.inst_eq(i);
                    self.op_name(*inst);
                    self.paren_start();
                    self.paren_end();
                }
                _ => unimplemented!("FIR op {:?} not implemented", inst),
            };
            let next_is_fun_end = matches!(self.fir.ops.get(i + 1), Some(Op::FunEnd));
            if next_is_fun_end {
                // set here so the newline is not printed with indent before closing brace
                let func_start = self.cur_func_i.take().expect("in function before func end");
                // NOTE: +3 for the +1 offset of func_start, funEnd, and ???
                self.offset += (i - func_start + 3) as u32;
            }
            let not_at_last_op = i < self.fir.ops.len();
            if not_at_last_op {
                self.newline();
            }
        }
        return self.str;
    }

    fn write(&mut self, str: &str) {
        self.str.push_str(str);
    }

    fn newline(&mut self) {
        self.write("\n");
        if self.in_func() {
            self.write(Self::INDENT);
        }
    }

    fn space(&mut self) {
        self.write(" ");
    }

    fn func_1<F>(&mut self, name: &str, arg: F)
    where
        F: FnOnce(&mut Self),
    {
        self.write(name);
        self.paren_start();
        arg(self);
        self.paren_end();
    }

    fn op_func_1<F>(&mut self, op: Op, arg: F)
    where
        F: FnOnce(&mut Self),
    {
        let name = self.get_op_name(op);
        self.func_1(name, arg);
    }

    fn op_func_1_ref(&mut self, op: &Op, arg: &Ref) {
        let name = self.get_op_name(*op);
        self.write(name);
        self.paren_start();
        self.write_ref(arg);
        self.paren_end();
    }

    fn op_func_1_ty(&mut self, op: &Op, arg: &TypeRef) {
        let name = self.get_op_name(*op);
        self.write(name);
        self.paren_start();
        self.write_type_ref(arg);
        self.paren_end();
    }

    fn func_2<F1, F2>(&mut self, name: &str, arg1: F1, arg2: F2)
    where
        F1: FnOnce(&mut Self),
        F2: FnOnce(&mut Self),
    {
        self.write(name);
        self.paren_start();
        arg1(self);
        self.sep();
        arg2(self);
        self.paren_end();
    }

    fn op_func_2<F1, F2>(&mut self, op: Op, arg1: F1, arg2: F2)
    where
        F1: FnOnce(&mut Self),
        F2: FnOnce(&mut Self),
    {
        let name = self.get_op_name(op);
        self.func_2(name, arg1, arg2);
    }
    fn op_func_2_ref_ref(&mut self, op: &Op, arg1: &Ref, arg2: &Ref) {
        let name = self.get_op_name(*op);
        self.write(name);
        self.paren_start();
        self.write_ref(arg1);
        self.sep();
        self.write_ref(arg2);
        self.paren_end();
    }

    fn op_func_2_ty_ref(&mut self, op: &Op, arg1: &TypeRef, arg2: &Ref) {
        let name = self.get_op_name(*op);
        self.write(name);
        self.paren_start();
        self.write_type_ref(arg1);
        self.sep();
        self.write_ref(arg2);
        self.paren_end();
    }

    fn op_func_3_ty_ref_ref(&mut self, op: &Op, ty: &TypeRef, a: &Ref, b: &Ref) {
        self.op_name(*op);
        self.paren_start();
        self.write_type_ref(ty);
        self.sep();
        self.write_ref(a);
        self.sep();
        self.write_ref(b);
        self.paren_end();
    }

    fn op_func_3_ref_ref_ref(&mut self, op: &Op, a: &Ref, b: &Ref, c: &Ref) {
        self.op_name(*op);
        self.paren_start();
        self.write_ref(a);
        self.sep();
        self.write_ref(b);
        self.sep();
        self.write_ref(c);
        self.paren_end();
    }

    fn inst_ref(&mut self, i: u32) {
        self.write("%");
        let i = i - self.offset;
        self.write(i.to_string().as_str());
    }

    fn inst_eq(&mut self, mut i: usize) {
        if let Some(func_offset) = self.cur_func_i {
            i -= func_offset;
        }
        self.inst_ref(i as u32);
        self.write(" = ");
    }

    fn func_ref(&mut self, i: DIndex) {
        self.write("@");
        self.write_ident(i);
    }

    fn write_ref(&mut self, r: &Ref) {
        match r {
            Ref::Const(i) => self.func_1("const", |s| s.write(&self.fir.get_const(*i).to_string())),
            Ref::Inst(i) => self.inst_ref(*i),
        }
    }

    fn write_type_ref(&mut self, ty: &TypeRef) {
        use TypeRef::*;
        let str = match ty {
            IntU64 => "u64",
            None => "_",
        };
        self.write(str);
    }

    fn write_type_ref_at(&mut self, i: usize) {
        let ty = &self.fir.types[i];
        self.write_type_ref(ty);
    }

    fn write_ident(&mut self, i: DIndex) {
        let str = self.fir.data.get_ref::<str>(i);
        self.write(str);
    }

    fn op_name(&mut self, op: Op) {
        let str = self.get_op_name(op);
        self.write(str);
    }

    fn in_func(&self) -> bool {
        return self.cur_func_i.is_some();
    }

    fn brack_start(&mut self) {
        self.write("[");
    }

    fn brack_end(&mut self) {
        self.write("]");
    }

    fn paren_start(&mut self) {
        self.write("(");
    }

    fn paren_end(&mut self) {
        self.write(")");
    }

    fn sep(&mut self) {
        self.write(", ");
    }
    fn get_op_name(&self, op: Op) -> &'static str {
        use Op::*;
        match op {
            FunDef { .. } => "define",
            FunEnd => "fun_end",
            FunArg => "arg",
            FunCall { .. } => "call",
            Ret(_) => "ret",
            Alloc => "alloc",
            Load(_) => "load",
            Store(_, _) => "store",
            Add(_, _) => "add",
            Sub(_, _) => "sub",
            Mul(_, _) => "mul",
            Div(_, _) => "div",
            Eq(_, _) => "cmp_eq",
            Lt(_, _) => "cmp_lt",
            LtEq(_, _) => "cmp_lteq",
            Gt(_, _) => "cmp_gt",
            GtEq(_, _) => "cmp_gteq",
            Branch { .. } => "br",
            Jump(_) => "jmp",
            Phi { .. } => "phi",
            Label => "label",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_fir_str_eq {
        ($contents:literal, $($lines:literal),*) => {
            let parser = crate::parser::Parser::new($contents);
            let ast = parser.parse().expect("syntax error");
            let fir = FIRGen::generate(ast).expect("failed to generate fir");
            let fir_str = FIRStringifier::stringify(&fir);
            let mut fir_lines = fir_str.lines();
            #[allow(unused)]
            let mut i = 0;
            $(
                #[allow(unused_assignments)]
                {
                    assert_eq!($lines, fir_lines.next().unwrap());
                    i += 1
                }
            )*
        };
    }

    #[test]
    fn add() {
        assert_fir_str_eq!(
            "( + 1 2)",
            "%0 = load(u64, const(1))",
            "%1 = load(u64, const(2))",
            "%2 = add(u64, %0, %1)"
        );
    }

    #[test]
    fn chained_algebra() {
        assert_fir_str_eq!(
            "(+ 0 ( + 1 ( * 2 (- 3 4))))",
            "%0 = load(u64, const(0))",
            "%1 = load(u64, const(1))",
            "%2 = load(u64, const(2))",
            "%3 = load(u64, const(3))",
            "%4 = load(u64, const(4))",
            "%5 = sub(u64, %3, %4)",
            "%6 = mul(u64, %2, %5)",
            "%7 = add(u64, %1, %6)",
            "%8 = add(u64, %0, %7)"
        );
    }

    #[test]
    fn if_expr() {
        assert_fir_str_eq!(
            "(if (== 1 2) 3 4)",
            "%0 = load(u64, const(1))",
            "%1 = load(u64, const(2))",
            "%2 = cmp_eq(u64, %0, %1)",
            "br(%2, %4, %7)",
            "%4 = label()",
            "%5 = load(u64, const(3))",
            "jmp(%10)",
            "%7 = label()",
            "%8 = load(u64, const(4))",
            "jmp(%10)",
            "%10 = label()",
            "%11 = phi(u64, [%4 -> %5], [%7 -> %8])"
        );
    }

    #[test]
    fn fundef() {
        assert_fir_str_eq!(
            "(fun add (a b) (+ a b))",
            "define u64 @add {",
            "  %0 = arg(u64)",
            "  %1 = arg(u64)",
            "  %2 = load(u64, %0)",
            "  %3 = load(u64, %1)",
            "  %4 = add(u64, %2, %3)",
            "  %5 = ret(%4)",
            "}"
        );
    }

    #[test]
    fn funcall() {
        assert_fir_str_eq!(
            "(fun add (a b) (+ a b)) (add 1 2)",
            "define u64 @add {",
            "  %0 = arg(u64)",
            "  %1 = arg(u64)",
            "  %2 = load(u64, %0)",
            "  %3 = load(u64, %1)",
            "  %4 = add(u64, %2, %3)",
            "  %5 = ret(%4)",
            "}",
            "%0 = load(u64, const(1))",
            "%1 = load(u64, const(2))",
            "%2 = call(u64, @add, [%0, %1])"
        );
    }

    #[test]
    fn bind() {
        assert_fir_str_eq!(
            "(let a 1)",
            "%0 = alloc(u64)",
            "%1 = load(u64, const(1))",
            "store(%0, %1)"
        );
    }

    #[test]
    fn bind_use() {
        assert_fir_str_eq!(
            "(let a 1) a",
            "%0 = alloc(u64)",
            "%1 = load(u64, const(1))",
            "store(%0, %1)",
            "%3 = load(u64, %0)"
        );
    }
}
