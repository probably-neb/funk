use crate::fir::*;

struct FirStr(String);

impl FirStr {
    fn write(&mut self, str: &str) {
        self.0.push_str(str);
    }
}

const INDENT: &'static str = "  ";

pub fn stringify(fir: &FIR) -> String {
    let mut _fs = FirStr(String::new());
    let fs = &mut _fs;

    let mut cur_func_i = None;
    let mut offset = 0;

    for (i, inst) in fir.ops.iter().enumerate() {
        match inst {
            Op::Load(r) => {
                write_inst_eq(fs, i, cur_func_i, offset);
                let ty = &fir.types[i];
                write_op_func_start(fs, inst);
                write_type_ref(fs, ty);
                write_sep(fs);
                write_ref(fs, r, fir, offset);
                write_func_end(fs);
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
                write_inst_eq(fs, i, cur_func_i, offset);
                let ty = &fir.types[i];
                write_op_name(fs, inst);
                write_paren_start(fs);
                write_type_ref(fs, ty);
                write_sep(fs);
                write_ref(fs, lhs, fir, offset);
                write_sep(fs);
                write_ref(fs, rhs, fir, offset);
                write_paren_end(fs);
            }
            Op::FunDef { name } => {
                // let in_func = in_func();
                // assert!(!in_func, "nested functions not supported");
                fs.write("define");
                write_space(fs);
                let return_ty = &fir.types[i];
                write_type_ref(fs, return_ty);
                write_space(fs);
                write_func_ref(fs, &fir.data, *name);
                write_space(fs);
                fs.write("{");
                cur_func_i = Some(i + 1);
            }
            Op::FunArg => {
                // assert!(in_func(), "arg decl outside of function");
                write_inst_eq(fs, i, cur_func_i, offset);
                let ty = &fir.types[i];
                write_op_func_1_ty(fs, inst, ty);
            }
            Op::FunCall { fun: fun_i, args } => {
                let Op::FunDef { name } = &fir.ops[*fun_i as usize] else {
                    unreachable!("fun call ref not fun def");
                };
                write_inst_eq(fs, i, cur_func_i, offset);
                write_op_name(fs, inst); // "call"
                write_paren_start(fs);
                write_type_ref_at(fs, &fir.types, i);
                write_sep(fs);
                write_func_ref(fs, &fir.data, *name);
                write_sep(fs);
                write_brack_start(fs);
                let ast::ExtraFunArgs { args } = fir.extra.get::<ast::ExtraFunArgs>(*args);
                for arg in args.iter().take(args.len() - 1) {
                    write_inst_ref(fs, *arg, offset);
                    write_sep(fs);
                }
                write_inst_ref(fs, *args.last().unwrap(), offset);
                write_brack_end(fs);
                write_paren_end(fs);
            }
            Op::Ret(r) => {
                write_inst_eq(fs, i, cur_func_i, offset);
                write_op_func_1_ref(fs, inst, r, &fir, offset);
            }
            Op::FunEnd => {
                fs.write("}");
            }
            Op::Alloc => {
                write_inst_eq(fs, i, cur_func_i, offset);
                let ty = &fir.types[i];
                write_op_func_1_ty(fs, inst, ty);
            }
            Op::Store(dest, src) => {
                write_op_func_start(fs, inst);
                write_ref(fs, dest, fir, offset);
                write_sep(fs);
                write_ref(fs, src, fir, offset);
                write_paren_end(fs);
            }
            Op::Branch { cond, t, f } => {
                write_op_name(fs, inst);
                write_paren_start(fs);
                write_ref(fs, cond, fir, offset);
                write_sep(fs);
                write_ref(fs, t, fir, offset);
                write_sep(fs);
                write_ref(fs, f, fir, offset);
                write_paren_end(fs);
            }
            Op::Jump(dest) => {
                write_op_func_1_ref(fs, inst, dest, &fir, offset);
            }
            Op::Phi {
                a: (a_from, a_res),
                b: (b_from, b_res),
            } => {
                write_inst_eq(fs, i, cur_func_i, offset);
                write_op_name(fs, inst);
                write_paren_start(fs);

                write_type_ref_at(fs, &fir.types, i);

                write_sep(fs);

                write_brack_start(fs);
                write_ref(fs, a_from, fir, offset);
                fs.write(" -> ");
                write_ref(fs, a_res, fir, offset);
                write_brack_end(fs);

                write_sep(fs);

                write_brack_start(fs);
                write_ref(fs, b_from, fir, offset);
                fs.write(" -> ");
                write_ref(fs, b_res, fir, offset);
                write_brack_end(fs);

                write_paren_end(fs);
            }
            Op::Label => {
                write_inst_eq(fs, i, cur_func_i, offset);
                write_op_name(fs, inst);
                write_paren_start(fs);
                write_paren_end(fs);
            }
        };
        let next_is_fun_end = matches!(fir.ops.get(i + 1), Some(Op::FunEnd));
        if next_is_fun_end {
            // set here so the newline is not printed with indent before closing brace
            let func_start = cur_func_i.take().expect("in function before func end");
            // NOTE: +3 for the +1 offset of func_start, funEnd, and ???
            offset += (i - func_start + 3) as u32;
        }
        let not_at_last_op = i < fir.ops.len();
        if not_at_last_op {
            write_newline(fs);
            if cur_func_i.is_some() {
                fs.write(INDENT);
            }
        }
    }
    return _fs.0;
}

fn write_op_func_1_ref(fs: &mut FirStr, op: &Op, arg: &Ref, fir: &FIR, offset: u32) {
    write_op_func_start(fs, op);
    write_ref(fs, arg, fir, offset);
    write_paren_end(fs);
}

fn write_op_func_1_ty(fs: &mut FirStr, op: &Op, arg: &TypeRef) {
    write_op_func_start(fs, op);
    write_type_ref(fs, arg);
    write_paren_end(fs);
}

fn write_func_ref(fs: &mut FirStr, data: &ast::DataPool, i: DIndex) {
    fs.write("@");
    write_ident(fs, data, i);
}

fn write_ref(fs: &mut FirStr, r: &Ref, fir: &FIR, offset: u32) {
    match r {
        Ref::Const(i) => {
            write_func_start(fs, "const");
            write_const(fs, fir, *i);
            write_func_end(fs);
        }
        Ref::Inst(i) => write_inst_ref(fs, *i, offset),
    }
}

fn write_op_func_start(fs: &mut FirStr, op: &Op) {
    let name = get_op_name(op);
    fs.write(name);
    write_paren_start(fs);
}

fn write_func_start(fs: &mut FirStr, name: &str) {
    fs.write(name);
    write_paren_start(fs);
}

fn write_func_end(fs: &mut FirStr) {
    write_paren_end(fs);
}

fn write_const(fs: &mut FirStr, fir: &FIR, i: u32) {
    let cnst = fir.get_const(i);
    fs.write(cnst.to_string().as_str());
}

fn write_inst_eq(fs: &mut FirStr, i: usize, cur_func_i: Option<usize>, offset: u32) {
    let mut inst_i = i;
    if let Some(func_offset) = cur_func_i {
        inst_i -= func_offset;
    }
    write_inst_ref(fs, inst_i as u32, offset);
    fs.write(" = ");
}

fn write_inst_ref(fs: &mut FirStr, i: u32, offset: u32) {
    fs.write("%");
    let ref_i = i - offset;
    fs.write(ref_i.to_string().as_str());
}
fn write_ident(fs: &mut FirStr, data: &ast::DataPool, i: DIndex) {
    let str = data.get_ref::<str>(i);
    fs.write(str);
}

fn write_type_ref(fs: &mut FirStr, ty: &TypeRef) {
    use TypeRef::*;
    let str = match ty {
        IntU64 => "u64",
        None => "_",
    };
    fs.write(str);
}

fn write_type_ref_at(fs: &mut FirStr, types: &[TypeRef], i: usize) {
    let ty = &types[i];
    write_type_ref(fs, ty);
}

fn write_newline(fs: &mut FirStr) {
    fs.write("\n");
}
fn write_space(fs: &mut FirStr) {
    fs.write(" ");
}

fn write_brack_start(fs: &mut FirStr) {
    fs.write("[");
}

fn write_brack_end(fs: &mut FirStr) {
    fs.write("]");
}

fn write_paren_start(fs: &mut FirStr) {
    fs.write("(");
}

fn write_paren_end(fs: &mut FirStr) {
    fs.write(")");
}

fn write_sep(fs: &mut FirStr) {
    fs.write(", ");
}

fn write_op_name(fs: &mut FirStr, op: &Op) {
    let str = get_op_name(op);
    fs.write(str);
}

fn get_op_name(op: &Op) -> &'static str {
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

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_fir_str_eq {
        ($contents:literal, $($lines:literal),*) => {
            let parser = crate::parser::Parser::new($contents);
            let ast = parser.parse().expect("syntax error");
            let fir = FIRGen::generate(ast).expect("failed to generate fir");
            let fir_str = stringify(&fir);
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
    fn all_binops() {
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
