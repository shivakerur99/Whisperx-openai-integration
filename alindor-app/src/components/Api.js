import axios from "axios";

const Api=axios.create({
    // baseURL: "http://localhost:8000", //for localhost connection with FastAPI backend
    // baseURL: "https://alindor-ev3t.onrender.com"

    // baseURL: "https://alindor-hm.onrender.com",
    baseURL: "https://shivakerur99-alindor-grandmaster.hf.space"
})

export default Api