const http = require("http");

const data = JSON.stringify({ query: "Chola chola tamil song" });

const options = {
  hostname: "127.0.0.1",  // ✅ This avoids IPv6 (::1)
  port: 8000,
  path: "/search",
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Content-Length": data.length
  }
};

const req = http.request(options, (res) => {
  let responseData = "";

  res.on("data", (chunk) => {
    responseData += chunk;
  });

  res.on("end", () => {
    try {
      const parsedData = JSON.parse(responseData);
      console.log("Search results:", parsedData);
    } catch (err) {
      console.error("Failed to parse response:", err);
    }
  });
});

req.on("error", (error) => {
  console.error("Request failed:", error);
});

req.write(data);
req.end();