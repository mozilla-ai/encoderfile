# Command-Line Help for `encoderfile`

This document contains the help content for the `encoderfile` command-line program.

**Command Overview:**

* [`encoderfile`↴](#encoderfile)
* [`encoderfile build`↴](#encoderfile-build)
* [`encoderfile version`↴](#encoderfile-version)

## `encoderfile`

**Usage:** `encoderfile <COMMAND>`

###### **Subcommands:**

* `build` — Build an encoderfile.
* `version` — Get Encoderfile version.



## `encoderfile build`

Build an encoderfile.

**Usage:** `encoderfile build [OPTIONS] -f <CONFIG>`

###### **Options:**

* `-f <CONFIG>` — Path to config file.
* `-o`, `--output-path <OUTPUT_PATH>` — Output path (e.g., `./my_model.encoderfile`)
* `--cache-dir <CACHE_DIR>` — Cache directory. This is used for build artifacts.
* `--no-build` — Skips build stage. Only generates files to directory in `cache_dir`.



## `encoderfile version`

Get Encoderfile version.

**Usage:** `encoderfile version`



<hr/>

<small><i>
    This document was generated automatically by
    <a href="https://crates.io/crates/clap-markdown"><code>clap-markdown</code></a>.
</i></small>
