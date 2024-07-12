
            /// Returns the `rustc` SemVer version and additional metadata
            /// like the git short hash and build date.
            pub fn version_meta() -> VersionMeta {
                VersionMeta {
                    semver: Version {
                        major: 1,
                        minor: 78,
                        patch: 0,
                        pre: vec![semver::Identifier::AlphaNumeric("nightly".to_owned()), ],
                        build: vec![],
                    },
                    host: "aarch64-apple-darwin".to_owned(),
                    short_version_string: "rustc 1.78.0-nightly (766bdce74 2024-03-16)".to_owned(),
                    commit_hash: Some("766bdce744d531267d53ba2a3f9ffcda69fb9b17".to_owned()),
                    commit_date: Some("2024-03-16".to_owned()),
                    build_date: None,
                    channel: Channel::Nightly,
                }
            }
            