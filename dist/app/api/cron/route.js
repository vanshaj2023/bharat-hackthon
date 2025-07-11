"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.GET = exports.revalidate = exports.dynamic = exports.maxDuration = void 0;
var server_1 = require("next/server");
var utils_1 = require("@/lib/utils");
var mongoose_1 = require("@/lib/mongoose");
var product_model_1 = __importDefault(require("@/lib/models/product.model"));
var scraper_1 = require("@/lib/scraper");
var nodemailer_1 = require("@/lib/nodemailer");
exports.maxDuration = 300; // This function can run for a maximum of 300 seconds
exports.dynamic = "force-dynamic";
exports.revalidate = 0;
function GET(request) {
    return __awaiter(this, void 0, void 0, function () {
        var products, updatedProducts, error_1;
        var _this = this;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 3, , 4]);
                    (0, mongoose_1.connectToDB)();
                    return [4 /*yield*/, product_model_1.default.find({})];
                case 1:
                    products = _a.sent();
                    if (!products)
                        throw new Error("No product fetched");
                    return [4 /*yield*/, Promise.all(products.map(function (currentProduct) { return __awaiter(_this, void 0, void 0, function () {
                            var scrapedProduct, updatedPriceHistory, product, updatedProduct, emailNotifType, productInfo, emailContent, userEmails;
                            return __generator(this, function (_a) {
                                switch (_a.label) {
                                    case 0: return [4 /*yield*/, (0, scraper_1.scrapeAmazonProduct)(currentProduct.url)];
                                    case 1:
                                        scrapedProduct = _a.sent();
                                        if (!scrapedProduct)
                                            return [2 /*return*/];
                                        updatedPriceHistory = __spreadArray(__spreadArray([], currentProduct.priceHistory, true), [
                                            {
                                                price: scrapedProduct.currentPrice,
                                            },
                                        ], false);
                                        product = __assign(__assign({}, scrapedProduct), { priceHistory: updatedPriceHistory, lowestPrice: (0, utils_1.getLowestPrice)(updatedPriceHistory), highestPrice: (0, utils_1.getHighestPrice)(updatedPriceHistory), averagePrice: (0, utils_1.getAveragePrice)(updatedPriceHistory) });
                                        return [4 /*yield*/, product_model_1.default.findOneAndUpdate({
                                                url: product.url,
                                            }, product)];
                                    case 2:
                                        updatedProduct = _a.sent();
                                        emailNotifType = (0, utils_1.getEmailNotifType)(scrapedProduct, currentProduct);
                                        if (!(emailNotifType && updatedProduct.users.length > 0)) return [3 /*break*/, 5];
                                        productInfo = {
                                            title: updatedProduct.title,
                                            url: updatedProduct.url,
                                        };
                                        return [4 /*yield*/, (0, nodemailer_1.generateEmailBody)(productInfo, emailNotifType)];
                                    case 3:
                                        emailContent = _a.sent();
                                        userEmails = updatedProduct.users.map(function (user) { return user.email; });
                                        // Send email notification
                                        return [4 /*yield*/, (0, nodemailer_1.sendEmail)(emailContent, userEmails)];
                                    case 4:
                                        // Send email notification
                                        _a.sent();
                                        _a.label = 5;
                                    case 5: return [2 /*return*/, updatedProduct];
                                }
                            });
                        }); }))];
                case 2:
                    updatedProducts = _a.sent();
                    return [2 /*return*/, server_1.NextResponse.json({
                            message: "Ok",
                            data: updatedProducts,
                        })];
                case 3:
                    error_1 = _a.sent();
                    throw new Error("Failed to get all products: ".concat(error_1.message));
                case 4: return [2 /*return*/];
            }
        });
    });
}
exports.GET = GET;
