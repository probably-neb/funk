Build system should be in the language

possibly per file:
```{funk}
// in `some_file_that_does_network_requests_with_libcurl_and_wget.fun`

let build = use("std").build

// build time dependency. Will panic if this file is required and compiler cannot find libcurl at buildtime
build.require.c_lib_exists("libcurl")
    .at(build.At.BuildTime)

// runtime requirement that `wget` is in the users path
build.require.binary_in_path("wget")
    .at(build.At.RunTime)
```

Having the build system in the standard library would also allow bindgen files
for creating wrappers around ffi to specify exactly how to build their dependencies
either at runtime or at buildtime

However bindgen is done therefore should support specifying a build script
