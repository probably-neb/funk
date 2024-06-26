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
    for expr_i in 0..ast.exprs.len() {
        if expr_i < visited {
            continue;
        }
        let node = expr_into_treenode(expr_i, ast, &mut visited);
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
        eprintln!(
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


fn expr_into_treenode(expr_i: usize, ast: &Ast, visited: &mut usize) -> TreeNode<String> {
    let expr = ast.exprs[expr_i];
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
        Expr::Nop | Expr::Int(_) | Expr::String(_) | Expr::Ident(_) | Expr::Bool(_) => {
            mark_visited(visited, expr_i);
        }
        Expr::FunCall { name, args, .. } => {
            for arg in ast.extra.iter(args) {
                let arg_node = expr_into_treenode(arg as usize,ast, visited);
                node.add_node(arg_node);
            }
        }
        Expr::If {
            cond,
            branch_then: branch_true,
            branch_else: branch_false,
        } => {
            let cond_node = expr_into_treenode(cond, ast, visited);
            node.add_node(cond_node);
            mark_visited(visited, cond);

            let mut true_node = TreeNode::new("Then".to_string());
            let mut last = branch_true;
            for (i, &expr) in ast.extra.indexed_iter_of(branch_true, &ast.exprs) {
                let expr_node = expr_into_treenode(i, ast, visited);
                true_node.add_node(expr_node);
                last = usize::max(i + 1,*visited);
            }
            node.add_node(true_node);
            mark_visited(visited, last);

            let mut false_node = TreeNode::new("Else".to_string());
            let mut last = branch_true;
            for (i, &expr) in ast.extra.indexed_iter_of(branch_false, &ast.exprs) {
                let expr_node = expr_into_treenode(i, ast, visited);
                false_node.add_node(expr_node);
                last = usize::max(i + 1,*visited);
            }
            node.add_node(false_node);
            mark_visited(visited, last);
        }
        Expr::Binop { lhs, rhs, .. } => {
            let lhs_node = expr_into_treenode(lhs, ast, visited);
            mark_visited(visited, lhs);
            node.add_node(lhs_node);

            let rhs_node = expr_into_treenode(rhs, ast, visited);
            mark_visited(visited, rhs);
            node.add_node(rhs_node);
        }
        Expr::FunDef { args, body, .. } => {
            let mut arg_node = TreeNode::new("Args".to_string());
            for arg_i in ast.extra.slice(args) {
                let arg = TreeNode::new(ast.get_ident(*arg_i as usize).to_string());
                arg_node.add_node(arg);
            }
            node.add_node(arg_node);

            // visiting body will skip the empty funarg nodes
            let mut body_node = TreeNode::new("Body".to_string());
            for (i, &expr) in ast.extra.indexed_iter_of(body, &ast.exprs) {
                assert_ne!(expr, Expr::FunArg);
                let body_expr_node = expr_into_treenode(i, ast, visited);
                body_node.add_node(body_expr_node);
                mark_visited(visited, i);
            }
            // mark_visited(visited, i + 1);
            node.add_node(body_node);
        }
        Expr::Bind { value, .. } => {
            let value = expr_into_treenode(value, ast, visited);
            node.add_node(value);
        }
        Expr::Return {value} => {
            if let Some(value) = value {
                let val = expr_into_treenode(usize::from(value), ast, visited);
                node.add_node(val);
            }
        }
        Expr::FunArg => unreachable!("fun arg"),
        Expr::While { cond, body } => {
            let mut cond_node = TreeNode::new("Cond".to_string());
            let cond_expr_node = expr_into_treenode(cond, ast, visited);
            cond_node.add_node(cond_expr_node);
            node.add_node(cond_node);
            mark_visited(visited, cond);
            let mut body_node = TreeNode::new("Body".to_string());
            let mut last = body;
            for (i, &expr) in ast.extra.indexed_iter_of(body, &ast.exprs) {
                let expr_node = expr_into_treenode(i, ast, visited);
                body_node.add_node(expr_node);
                last = usize::max(i + 1,*visited);
            }
            node.add_node(body_node);
            mark_visited(visited, last);
        },
        Expr::Assign {value, ..} => {
            let value = expr_into_treenode(value, ast, visited);
            node.add_node(value);
        }
        Expr::Print {value} => {
            let value = expr_into_treenode(value, ast, visited);
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
        Expr::FunDef { name, .. } => format!("Fun {}", ast.get_ident(name)),
        Expr::FunCall { name, .. } => format!("Call {}", ast.get_ident(name)),
        Expr::Ident(i) => format!("Ident {}", ast.get_ident(i)),
        Expr::String(i) => format!("String \"{}\"", ast.get_ident(i)),
        Expr::Bool(value) => format!("Bool {}", value),
        Expr::Return{..} => format!("Return"),
        Expr::Bind { name, .. } => format!("let {}", ast.get_ident(name)),
        Expr::Assign { name, .. } => format!("assign {}", ast.get_ident(name)),
        Expr::FunArg => unreachable!("tried to format fun arg"),
        Expr::While {..} => "While".to_string(),
        Expr::Print { .. } => "Print".to_string(),
    }
}


fn mark_visited(cursor: &mut usize, i: usize) {
    *cursor = usize::max(*cursor, i + 1);
}
