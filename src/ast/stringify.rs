use std::cell::RefCell;
use std::rc::Rc;

use crate::ast::Ast;
use crate::parser::Expr;


pub fn print_tree(ast: &Ast) {
    let tree = into_tree(ast);
    tree.print("".to_string(), false);
}

fn into_tree(ast: &Ast) -> TreeNode<String> {
    let mut visited = vec![false; ast.exprs.len()];
    let mut root = TreeNode::new("Root".to_string());
    for (i, expr) in ast.exprs.iter().enumerate() {
        if !visited[i] {
            let node = expr_into_treenode(expr.clone(), ast, &mut visited);
            root.add_node(node);
        }
    }
    return root;
}

#[derive(Debug)]
struct TreeNode<T: std::fmt::Display> {
    data: T,
    children: Vec<Rc<RefCell<TreeNode<T>>>>,
}

impl<T: std::fmt::Display> TreeNode<T> {
    // Create a new tree node
    fn new(data: T) -> Self {
        TreeNode {
            data,
            children: vec![],
        }
    }

    // Add a child node directly to this node
    fn add_node(&mut self, data: TreeNode<T>) {
        self.children.push(Rc::new(RefCell::new(data)));
    }

    // Print the tree
    fn print(&self, prefix: String, is_last: bool) {
        println!(
            "{}{}{}",
            prefix,
            if is_last { "└─ " } else { "├─ " },
            self.data
        );
        let new_prefix = if is_last { "    " } else { "|   " };

        let children = &self.children;
        let last_index = children.len().saturating_sub(1);

        for (index, child) in children.iter().enumerate() {
            TreeNode::print(
                &child.borrow(),
                format!("{}{}", prefix, new_prefix),
                index == last_index,
            );
        }
    }
}


fn expr_into_treenode(expr: Expr, ast: &Ast, visited: &mut [bool]) -> TreeNode<String> {
    let data = repr_expr(expr, ast);
    let mut node = TreeNode::new(data);
    macro_rules! visit {
        ($node:ident, $i:expr) => {
            visited[$i] = true;
            let child = expr_into_treenode(ast.exprs[$i], ast, visited);
            $node.add_node(child);
        };
        ($i:expr) => {
            visit!(node, $i);
        };
    }
    use Expr::*;
    match expr {
        Nop | Int(_) | String(_) | Ident(_) => {}
        FunCall { args, .. } => {
            let names: Vec<&str> = ast.fun_arg_names_iter(args).collect();
            node.add_node(TreeNode::new(format!("args: [{}]", names.join(", "))));
        }
        If {
            cond,
            branch_true,
            branch_false,
        } => {
            visit!(cond);
            visit!(branch_true);
            visit!(branch_false);
        }
        Binop { lhs, rhs, .. } => {
            visit!(lhs);
            visit!(rhs);
        }
        FunDef { args, body, .. } => {
            let mut arg_node = TreeNode::new("Args".to_string());
            for arg_i in ast.fun_args_slice(args) {
                let arg = expr_into_treenode(expr, ast, visited);
                arg_node.add_node(arg);
                visited[*arg_i as usize] = true;
            }
            node.add_node(arg_node);

            visit!(body);
        }
        Bind { name, value } => {
            let name = expr_into_treenode(ast.exprs[name], ast, visited);
            let value = expr_into_treenode(ast.exprs[value], ast, visited);
            node.add_node(name);
            node.add_node(value);
        }
    }
    return node;
}

fn repr_expr(expr: Expr, ast: &Ast) -> String {
    match expr {
        Expr::Nop => "Nop".to_string(),
        Expr::Int(i) => format!("Int {}", ast.data.get::<u64>(i)),
        Expr::Binop { op, .. } => format!("{:?}", op),
        Expr::If { .. } => "If".to_string(),
        Expr::FunDef { name, .. } => format!("Fun {:?}", ast.get_ident(name)),
        Expr::FunCall { name, .. } => format!("Call {:?}", ast.get_ident(name)),
        Expr::Ident(i) => format!("Ident {:?}", ast.get_ident(i)),
        Expr::String(i) => format!("Str \"{:?}\"", ast.get_ident(i)),
        Expr::Bind { name, .. } => format!("let {:?}", ast.get_ident(name)),
    }
}
