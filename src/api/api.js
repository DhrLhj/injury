import request from "@/utils/request";


export const sendPhoto = (params) =>
  request({
    // url: "/file/uplo",
    url:"/classify",
    method: "post",
    data: params,
    headers: {
        'Content-Type': "multipart/form-data;"
    },
  });