use std::cell::RefCell;
use std::rc::Rc;

use crate::ast::Ast;
use crate::parser::Expr;


pub fn print_tree(ast: &Ast) {
    let tree = into_tree(ast);
    for (i, child) in tree.children.iter().enumerate() {
        let is_last = i == tree.children.len() - 1;
        child.borrow().print("".to_string(), is_last);
    }
}

fn into_tree(ast: &Ast) -> TreeNode<String> {
    let mut visited = 0;
    let mut root = TreeNode::new("Root".to_string());
    for (i, expr) in ast.exprs.iter().enumerate() {
        if i < visited {
            continue;
        }
        let node = expr_into_treenode(expr.clone(), ast, &mut visited);
        root.add_node(node);
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
        let new_prefix = if is_last { "    " } else { "│   " };

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


fn expr_into_treenode(expr: Expr, ast: &Ast, visited: &mut usize) -> TreeNode<String> {
    let data = repr_expr(expr, ast);
    let mut node = TreeNode::new(data);
    macro_rules! visit {
        ($node:ident, $i:expr) => {
            mark_visited(visited, $i);
            let child = expr_into_treenode(ast.exprs[$i], ast, visited);
            $node.add_node(child);
        };
        ($i:expr) => {
            visit!(node, $i);
        };
    }
    use Expr::*;
    match expr {
        Expr::Nop | Expr::Int(_) | Expr::String(_) | Expr::Ident(_) => {
        }
        Expr::FunCall { args, .. } => {
            let names: Vec<&str> = ast.fun_arg_names_iter(args).collect();
            node.add_node(TreeNode::new(format!("args: [{}]", names.join(", "))));
        }
        Expr::If {
            cond,
            branch_true,
            branch_false,
        } => {
            let cond_node = expr_into_treenode(ast.exprs[cond], ast, visited);
            node.add_node(cond_node);
            mark_visited(visited, cond);

            let mut true_node = TreeNode::new("Then".to_string());
            let (mut i, end) = branch_true.tup_i();
            while i != end {
                let expr_node = expr_into_treenode(ast.exprs[i], ast, visited);
                true_node.add_node(expr_node);
                i = usize::max(i + 1,*visited);
            }
            node.add_node(true_node);
            mark_visited(visited, i);

            let mut false_node = TreeNode::new("Else".to_string());
            let (mut i, end) = branch_false.tup_i();
            while i != end {
                let expr_node = expr_into_treenode(ast.exprs[i], ast, visited);
                false_node.add_node(expr_node);
                i = usize::max(i + 1,*visited);
            }
            node.add_node(false_node);
            mark_visited(visited, i);
        }
        Expr::Binop { lhs, rhs, .. } => {
            let lhs_node = expr_into_treenode(ast.exprs[lhs], ast, visited);
            mark_visited(visited, lhs);
            node.add_node(lhs_node);

            let rhs_node = expr_into_treenode(ast.exprs[rhs], ast, visited);
            mark_visited(visited, rhs);
            node.add_node(rhs_node);
        }
        Expr::FunDef { args, body, .. } => {
            let mut arg_node = TreeNode::new("Args".to_string());
            for arg_i in ast.fun_args_slice(args) {
                let arg = TreeNode::new(ast.get_ident(*arg_i as usize).to_string());
                arg_node.add_node(arg);
            }
            node.add_node(arg_node);

            // visiting body will skip the empty funarg nodes
            let mut body_node = TreeNode::new("Body".to_string());
            let (mut i, end) = body.tup_i();
            while i != end {
                assert_ne!(ast.exprs[i], Expr::FunArg);
                let body_expr_node = expr_into_treenode(ast.exprs[i], ast, visited);
                body_node.add_node(body_expr_node);
                i = usize::max(i + 1,*visited);
            }
            mark_visited(visited, i + 1);
            node.add_node(body_node);
        }
        Expr::Bind { name, value } => {
            let name = expr_into_treenode(ast.exprs[name], ast, visited);
            let value = expr_into_treenode(ast.exprs[value], ast, visited);
            node.add_node(name);
            node.add_node(value);
        }
        Expr::FunArg => unreachable!("fun arg"),
    }
    return node;
}

fn repr_expr(expr: Expr, ast: &Ast) -> String {
    match expr {
        Expr::Nop => "Nop".to_string(),
        Expr::Int(i) => format!("Int {}", ast.data.get::<u64>(i)),
        Expr::Binop { op, .. } => format!("{:?}", op),
        Expr::If { .. } => "If".to_string(),
        Expr::FunDef { name, .. } => format!("Fun {}", ast.get_ident(name)),
        Expr::FunCall { name, .. } => format!("Call {}", ast.get_ident(name)),
        Expr::Ident(i) => format!("Ident {}", ast.get_ident(i)),
        Expr::String(i) => format!("Str \"{}\"", ast.get_ident(i)),
        Expr::Bind { name, .. } => format!("let {}", ast.get_ident(name)),
        Expr::FunArg => unreachable!("tried to format fun arg"),
    }
}


fn mark_visited(cursor: &mut usize, i: usize) {
    *cursor = usize::max(*cursor, i + 1);
}
