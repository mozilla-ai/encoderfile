macro_rules! terminal_print_fn {
    // ----- single message -----
    ($fn_name:ident, $msg:ident => $body:block) => {
        pub fn $fn_name($msg: impl std::fmt::Display) {
            #[cfg(feature = "cli")]
            {
                use console::style;
                $body
            }

            #[cfg(not(feature = "cli"))]
            {
                let _ = &$msg;
            }
        }
    };

    // ----- key/value -----
    ($fn_name:ident, $key:ident, $value:ident => $body:block) => {
        pub fn $fn_name($key: impl std::fmt::Display, $value: impl std::fmt::Display) {
            #[cfg(feature = "cli")]
            {
                use console::style;
                $body
            }

            #[cfg(not(feature = "cli"))]
            {
                let _ = (&$key, &$value);
            }
        }
    };
}

terminal_print_fn! {
    success,
    msg => {
        println!("{} {}", style("✓").green(), msg);
    }
}

terminal_print_fn! {
    info,
    msg => {
        println!("{} {}", style("•").blue(), msg);
    }
}

terminal_print_fn! {
    warn,
    msg => {
        eprintln!("{} {}", style("!").yellow(), msg);
    }
}

terminal_print_fn! {
    error,
    msg => {
        eprintln!("{} {}", style("✗").red(), style(msg).bold());
    }
}

terminal_print_fn!(success_kv, key, value => {
    println!(
        "{} {} {}",
        style("✓").green(),
        style(key).dim(),
        style(value).bold(),
    );
});

terminal_print_fn!(info_kv, key, value => {
    println!(
        "{} {} {}",
        style("•").blue(),
        style(key).dim(),
        style(value).bold(),
    );
});

terminal_print_fn!(warn_kv, key, value => {
    eprintln!(
        "{} {} {}",
        style("!").yellow(),
        style(key).dim(),
        style(value).bold(),
    );
});

terminal_print_fn!(error_kv, key, value => {
    eprintln!(
        "{} {} {}",
        style("✗").red(),
        style(key).dim(),
        style(value).bold(),
    );
});
